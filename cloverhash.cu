extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
#include "nightcap/nightcap.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint64_t *d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

typedef union _NCMixNode {
	uint32_t values[16];
	uint4 nodes4[4];
} NCMixNode;

typedef union _NCLightNode {
	uint32_t values[8];
	uint4 nodes4[2];
} NCLightNode;

// statics
__constant__ uint32_t nc_d_dag_size;
__constant__ NCMixNode* nc_d_dag;
__constant__ uint32_t nc_d_light_size;
__constant__ NCLightNode* nc_d_light;
__constant__ uint32_t nc_d_height;


// State per GPU
typedef struct _NCGPUState
{
	NCLightNode* cache_nodes;
	NCMixNode* dag_nodes;

	uint64_t dag_size;
	uint64_t cache_size;

	uint64_t num_dag_nodes;
	uint64_t num_cache_nodes;

	uint32_t epoch;
	uint32_t height;
} NCGPUState;


// State per thread (to keep track of when we need to reset cuda vars)
typedef struct _NCThreadState
{
	uint32_t epoch;
} NCThreadState;


static NCGPUState nc_gpu_state[MAX_GPUS];
static NCThreadState nc_thread_state[MAX_GPUS];

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);

extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void skein256_cpu_init(int thr_id, uint32_t threads);
extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t* d_matrix);

extern void bmw256_setTarget(const void *ptarget);
extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_free(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces);
extern void bmw256_cpu_hash_32_to_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order);

__device__ __inline__ uint32_t fnv(const uint32_t v1, const uint32_t v2) {
 return (((v1 * NIGHTCAP_FNV_PRIME) ^ v2) % (0xFFFFFFFF));
}

__device__ __inline__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = (a.x * NIGHTCAP_FNV_PRIME ^ b.x) % 0xFFFFFFFF;
	c.y = (a.y * NIGHTCAP_FNV_PRIME ^ b.y) % 0xFFFFFFFF;
	c.z = (a.z * NIGHTCAP_FNV_PRIME ^ b.z) % 0xFFFFFFFF;
	c.w = (a.w * NIGHTCAP_FNV_PRIME ^ b.w) % 0xFFFFFFFF;
	return c;
}

__device__ __inline__ uint4 fnv4_int(uint4 a, uint32_t b)
{
	uint4 c;
	c.x = (a.x * NIGHTCAP_FNV_PRIME ^ b) % 0xFFFFFFFF;
	c.y = (a.y * NIGHTCAP_FNV_PRIME ^ b) % 0xFFFFFFFF;
	c.z = (a.z * NIGHTCAP_FNV_PRIME ^ b) % 0xFFFFFFFF;
	c.w = (a.w * NIGHTCAP_FNV_PRIME ^ b) % 0xFFFFFFFF;
	return c;
}

__device__ __inline__ uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

__device__ __inline__ void hashimoto_mix(uint32_t* headerHash, uint32_t* mixhash)
{
	NCMixNode mix;

	#pragma unroll 2
	for (int i=0; i<2; i++)
	{
		mix.values[i*8] = headerHash[0];
		mix.values[(i*8)+1] = headerHash[1];
		mix.values[(i*8)+2] = headerHash[2];
		mix.values[(i*8)+3] = headerHash[3];
		mix.values[(i*8)+4] = headerHash[4];
		mix.values[(i*8)+5] = headerHash[5];
		mix.values[(i*8)+6] = headerHash[6];
		mix.values[(i*8)+7] = headerHash[7];
	}

	uint32_t header_int = mix.values[0];

	for (uint32_t i = 0; i < 64; i++) {
		const uint32_t p = fnv(i ^ header_int, mix.values[i % 16]) % (nc_d_dag_size / 2);
		mix.nodes4[0] = fnv4(mix.nodes4[0], nc_d_dag[p].nodes4[0]);
		mix.nodes4[1] = fnv4(mix.nodes4[1], nc_d_dag[p].nodes4[1]);
		mix.nodes4[2] = fnv4(mix.nodes4[2], nc_d_dag[p].nodes4[2]);
		mix.nodes4[3] = fnv4(mix.nodes4[3], nc_d_dag[p].nodes4[3]);
	}

	mixhash[0] = fnv_reduce(mix.nodes4[0]);
	mixhash[1] = fnv_reduce(mix.nodes4[1]);
	mixhash[2] = fnv_reduce(mix.nodes4[2]);
	mixhash[3] = fnv_reduce(mix.nodes4[3]);
}

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
						}

__device__ __inline__ uint2 ROR8(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x0765);
	result.y = __byte_perm(a.x, a.y, 0x0765);
	return result;
}


__constant__ uint2 keccak_round_constants35[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

static void __forceinline__ __device__ keccak_block(uint2 *s)
{
	uint2 bc[5], tmpxor[5], u, v;
	//	uint2 s[25];

	#pragma unroll 1
	for (int i = 0; i < 24; i++)
	{
		#pragma unroll
		for (uint32_t x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		u = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(u, 1);

		u = s[0]; v = s[1]; s[0] ^= (~v) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & u; s[4] ^= (~u) & v;
		u = s[5]; v = s[6]; s[5] ^= (~v) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & u; s[9] ^= (~u) & v;
		u = s[10]; v = s[11]; s[10] ^= (~v) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & u; s[14] ^= (~u) & v;
		u = s[15]; v = s[16]; s[15] ^= (~v) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & u; s[19] ^= (~u) & v;
		u = s[20]; v = s[21]; s[20] ^= (~v) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & u; s[24] ^= (~u) & v;
		s[0] ^= keccak_round_constants35[i];
	}
}

//__launch_bounds__(256)
__global__
void nightcap_gpu_hash_52(const uint32_t threads, const uint32_t startNonce, uint64_t * Hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t T0 = 0x1a0U;
		uint32_t v[16];

		const uint32_t  u256[16] = {
			0x243F6A88, 0x85A308D3,
			0x13198A2E, 0x03707344,
			0xA4093822, 0x299F31D0,
			0x082EFA98, 0xEC4E6C89,
			0x452821E6, 0x38D01377,
			0xBE5466CF, 0x34E90C6C,
			0xC0AC29B7, 0xC97C50DD,
			0x3F84D5B5, 0xB5470917
		};

		uint32_t m[16];

		LOHI(m[0], m[1], __ldg(&((uint64_t*)Hash)[thread]));
		LOHI(m[2], m[3], __ldg(&((uint64_t*)Hash)[thread + 1 * threads]));
		LOHI(m[4], m[5], __ldg(&((uint64_t*)Hash)[thread + 2 * threads]));
		LOHI(m[6], m[7], __ldg(&((uint64_t*)Hash)[thread + 3 * threads]));

		m[8] = nc_d_height;

		// mix mix mix
		hashimoto_mix(m, &m[9]);

		#pragma unroll 8
		for (uint32_t i=0; i<13; i++)
		{
			m[i] = cuda_swab32(m[i]);
		}

		// padding
		m[13] = 2147483649;
		m[14] = 0;
		m[15] = 416;

		v[0] = ((uint32_t)(0x6a09e667U)); 
		v[1] = ((uint32_t)(0xbb67ae85U)); 
		v[2] = ((uint32_t)(0x3c6ef372U)); 
		v[3] = ((uint32_t)(0xa54ff53aU)); 
		v[4] = ((uint32_t)(0x510e527fU)); 
		v[5] = ((uint32_t)(0x9b05688cU)); 
		v[6] = ((uint32_t)(0x1f83d9abU)); 
		v[7] = ((uint32_t)(0x5be0cd19U)); 

		v[8] = u256[0];
		v[9] = u256[1];
		v[10] = u256[2];
		v[11] = u256[3];
		v[12] = u256[4] ^ T0;
		v[13] = u256[5] ^ T0;
		v[14] = u256[6];
		v[15] = u256[7];

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);
		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);
		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);
		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		GSPREC(0, 4, 0x8, 0xC, 6, 15);
		GSPREC(1, 5, 0x9, 0xD, 14, 9);
		GSPREC(2, 6, 0xA, 0xE, 11, 3);
		GSPREC(3, 7, 0xB, 0xF, 0, 8);
		GSPREC(0, 5, 0xA, 0xF, 12, 2);
		GSPREC(1, 6, 0xB, 0xC, 13, 7);
		GSPREC(2, 7, 0x8, 0xD, 1, 4);
		GSPREC(3, 4, 0x9, 0xE, 10, 5);
		//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		GSPREC(0, 4, 0x8, 0xC, 10, 2);
		GSPREC(1, 5, 0x9, 0xD, 8, 4);
		GSPREC(2, 6, 0xA, 0xE, 7, 6);
		GSPREC(3, 7, 0xB, 0xF, 1, 5);
		GSPREC(0, 5, 0xA, 0xF, 15, 11);
		GSPREC(1, 6, 0xB, 0xC, 9, 14);
		GSPREC(2, 7, 0x8, 0xD, 3, 12);
		GSPREC(3, 4, 0x9, 0xE, 13, 0);
		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		uint32_t h[8];

		h[0] = cuda_swab32(0x6a09e667U ^ v[0] ^ v[8]);
		h[1] = cuda_swab32(0xbb67ae85U ^ v[1] ^ v[9]);
		h[2] = cuda_swab32(0x3c6ef372U ^ v[2] ^ v[10]);
		h[3] = cuda_swab32(0xa54ff53aU ^ v[3] ^ v[11]);
		h[4] = cuda_swab32(0x510e527fU ^ v[4] ^ v[12]);
		h[5] = cuda_swab32(0x9b05688cU ^ v[5] ^ v[13]);
		h[6] = cuda_swab32(0x1f83d9abU ^ v[6] ^ v[14]);
		h[7] = cuda_swab32(0x5be0cd19U ^ v[7] ^ v[15]);

		uint2 keccak_gpu_state[25] = { 0 };
		keccak_gpu_state[0].x = h[0];
		keccak_gpu_state[0].y = h[1];
		keccak_gpu_state[1].x = h[2];
		keccak_gpu_state[1].y = h[3];
		keccak_gpu_state[2].x = h[4];
		keccak_gpu_state[2].y = h[5];
		keccak_gpu_state[3].x = h[6];
		keccak_gpu_state[3].y = h[7];
		keccak_gpu_state[4] = make_uint2(1, 0);

		keccak_gpu_state[16] = make_uint2(0, 0x80000000);
		keccak_block(keccak_gpu_state);

		uint64_t *outputHash = (uint64_t *)Hash;
		#pragma unroll 4
		for (int i = 0; i<4; i++)
			outputHash[i*threads + thread] = devectorize(keccak_gpu_state[i]);
	}
}


__host__
void nightcap_blakeKeccak_hashimoto_cpu_hash_32(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	nightcap_gpu_hash_52 <<<grid, block>>> (threads, startNonce, Hash);
}

__global__
void nightcap_recalc_dag_item(const uint32_t start)
{
	uint32_t NodeIdx = start + (blockIdx.x * blockDim.x + threadIdx.x);

	if (NodeIdx >= nc_d_dag_size)
	return;

	NCLightNode DAGNode = nc_d_light[NodeIdx % nc_d_light_size];

	DAGNode.values[0] ^= NodeIdx;

	const uint32_t T0 = ((0xFFFFFE00) + 256) + 512;
	const uint32_t  u256[16] = {
		0x243F6A88, 0x85A308D3,
		0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0,
		0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377,
		0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD,
		0x3F84D5B5, 0xB5470917
	};

	// First blake round
	{
		uint32_t v[16];
		uint32_t m[16];

		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			m[i] = cuda_swab32(DAGNode.values[i]);
		}

		// padding
		m[8] = 2147483648;
		m[9] = 0;
		m[10] = 0;
		m[11] = 0;
		m[12] = 0;
		m[13] = 1;
		m[14] = 0;
		m[15] = 256;

		v[0] = ((uint32_t)(0x6a09e667U)); 
		v[1] = ((uint32_t)(0xbb67ae85U)); 
		v[2] = ((uint32_t)(0x3c6ef372U)); 
		v[3] = ((uint32_t)(0xa54ff53aU)); 
		v[4] = ((uint32_t)(0x510e527fU)); 
		v[5] = ((uint32_t)(0x9b05688cU)); 
		v[6] = ((uint32_t)(0x1f83d9abU)); 
		v[7] = ((uint32_t)(0x5be0cd19U)); 

		v[8] = u256[0];
		v[9] = u256[1];
		v[10] = u256[2];
		v[11] = u256[3];
		v[12] = u256[4] ^ T0;
		v[13] = u256[5] ^ T0;
		v[14] = u256[6];
		v[15] = u256[7];

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);
		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);
		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);
		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		GSPREC(0, 4, 0x8, 0xC, 6, 15);
		GSPREC(1, 5, 0x9, 0xD, 14, 9);
		GSPREC(2, 6, 0xA, 0xE, 11, 3);
		GSPREC(3, 7, 0xB, 0xF, 0, 8);
		GSPREC(0, 5, 0xA, 0xF, 12, 2);
		GSPREC(1, 6, 0xB, 0xC, 13, 7);
		GSPREC(2, 7, 0x8, 0xD, 1, 4);
		GSPREC(3, 4, 0x9, 0xE, 10, 5);
		//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		GSPREC(0, 4, 0x8, 0xC, 10, 2);
		GSPREC(1, 5, 0x9, 0xD, 8, 4);
		GSPREC(2, 6, 0xA, 0xE, 7, 6);
		GSPREC(3, 7, 0xB, 0xF, 1, 5);
		GSPREC(0, 5, 0xA, 0xF, 15, 11);
		GSPREC(1, 6, 0xB, 0xC, 9, 14);
		GSPREC(2, 7, 0x8, 0xD, 3, 12);
		GSPREC(3, 4, 0x9, 0xE, 13, 0);
		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		DAGNode.values[0] = cuda_swab32(0x6a09e667U ^ v[0] ^ v[8]);
		DAGNode.values[1] = cuda_swab32(0xbb67ae85U ^ v[1] ^ v[9]);
		DAGNode.values[2] = cuda_swab32(0x3c6ef372U ^ v[2] ^ v[10]);
		DAGNode.values[3] = cuda_swab32(0xa54ff53aU ^ v[3] ^ v[11]);
		DAGNode.values[4] = cuda_swab32(0x510e527fU ^ v[4] ^ v[12]);
		DAGNode.values[5] = cuda_swab32(0x9b05688cU ^ v[5] ^ v[13]);
		DAGNode.values[6] = cuda_swab32(0x1f83d9abU ^ v[6] ^ v[14]);
		DAGNode.values[7] = cuda_swab32(0x5be0cd19U ^ v[7] ^ v[15]);
	}

	for (uint32_t parent = 0; parent < NIGHTCAP_DATASET_PARENTS; ++parent)
	{
		// Calculate parent
		uint32_t ParentIdx = fnv(NodeIdx ^ parent, DAGNode.values[parent & 7]) % nc_d_light_size;
		const NCLightNode *ParentNode = nc_d_light + ParentIdx;

		DAGNode.nodes4[0] = fnv4_int(DAGNode.nodes4[0], ParentNode->values[0]);
		DAGNode.nodes4[1] = fnv4_int(DAGNode.nodes4[1], ParentNode->values[0]);
	}

	
	// Last blake round
	{
		uint32_t v[16];
		uint32_t m[16];

		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			m[i] = cuda_swab32(DAGNode.values[i]);
		}

		// padding
		m[8] = 2147483648;
		m[9] = 0;
		m[10] = 0;
		m[11] = 0;
		m[12] = 0;
		m[13] = 1;
		m[14] = 0;
		m[15] = 256;

		v[0] = ((uint32_t)(0x6a09e667U)); 
		v[1] = ((uint32_t)(0xbb67ae85U)); 
		v[2] = ((uint32_t)(0x3c6ef372U)); 
		v[3] = ((uint32_t)(0xa54ff53aU)); 
		v[4] = ((uint32_t)(0x510e527fU)); 
		v[5] = ((uint32_t)(0x9b05688cU)); 
		v[6] = ((uint32_t)(0x1f83d9abU)); 
		v[7] = ((uint32_t)(0x5be0cd19U)); 

		v[8] = u256[0];
		v[9] = u256[1];
		v[10] = u256[2];
		v[11] = u256[3];
		v[12] = u256[4] ^ T0;
		v[13] = u256[5] ^ T0;
		v[14] = u256[6];
		v[15] = u256[7];

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);
		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);
		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);
		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		GSPREC(0, 4, 0x8, 0xC, 6, 15);
		GSPREC(1, 5, 0x9, 0xD, 14, 9);
		GSPREC(2, 6, 0xA, 0xE, 11, 3);
		GSPREC(3, 7, 0xB, 0xF, 0, 8);
		GSPREC(0, 5, 0xA, 0xF, 12, 2);
		GSPREC(1, 6, 0xB, 0xC, 13, 7);
		GSPREC(2, 7, 0x8, 0xD, 1, 4);
		GSPREC(3, 4, 0x9, 0xE, 10, 5);
		//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		GSPREC(0, 4, 0x8, 0xC, 10, 2);
		GSPREC(1, 5, 0x9, 0xD, 8, 4);
		GSPREC(2, 6, 0xA, 0xE, 7, 6);
		GSPREC(3, 7, 0xB, 0xF, 1, 5);
		GSPREC(0, 5, 0xA, 0xF, 15, 11);
		GSPREC(1, 6, 0xB, 0xC, 9, 14);
		GSPREC(2, 7, 0x8, 0xD, 3, 12);
		GSPREC(3, 4, 0x9, 0xE, 13, 0);
		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		DAGNode.values[0] = cuda_swab32(0x6a09e667U ^ v[0] ^ v[8]);
		DAGNode.values[1] = cuda_swab32(0xbb67ae85U ^ v[1] ^ v[9]);
		DAGNode.values[2] = cuda_swab32(0x3c6ef372U ^ v[2] ^ v[10]);
		DAGNode.values[3] = cuda_swab32(0xa54ff53aU ^ v[3] ^ v[11]);
		DAGNode.values[4] = cuda_swab32(0x510e527fU ^ v[4] ^ v[12]);
		DAGNode.values[5] = cuda_swab32(0x9b05688cU ^ v[5] ^ v[13]);
		DAGNode.values[6] = cuda_swab32(0x1f83d9abU ^ v[6] ^ v[14]);
		DAGNode.values[7] = cuda_swab32(0x5be0cd19U ^ v[7] ^ v[15]);
	}

	((NCLightNode*)nc_d_dag)[NodeIdx] = DAGNode;
}

__host__
void nightcap_recalc_dag(
	uint64_t dag_nodes,
	uint32_t threads
	)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	uint32_t const work = grid.x * block.x;
	uint32_t const fullRuns = dag_nodes / (work);
	uint32_t const restWork = dag_nodes % (work);

	// Normal runs
	for (uint32_t i = 0; i < fullRuns; i++)
	{
		//printf("NC RUN: %u\n", i * work);
		nightcap_recalc_dag_item <<<grid, block, 0>>>(i * work);

	}
	cudaDeviceSynchronize();

	// Final run
	if (restWork > 0)
	{
	//printf("NC FINAL RUN: %u\n", fullRuns * work);
		nightcap_recalc_dag_item <<<grid, block, 0>>>(fullRuns * work);
	}

	cudaDeviceSynchronize();

	cudaGetLastError();
}



__host__
void nightcap_set_mix_constants(
	NCMixNode* _dag,
	uint32_t _dag_size,
	NCLightNode * _light,
	uint32_t _light_size,
	uint32_t _height
	)
{
	cudaMemcpyToSymbol(nc_d_dag, &_dag, sizeof(NCMixNode *));
	cudaMemcpyToSymbol(nc_d_dag_size, &_dag_size, sizeof(uint32_t));
	cudaMemcpyToSymbol(nc_d_light, &_light, sizeof(NCLightNode *));
	cudaMemcpyToSymbol(nc_d_light_size, &_light_size, sizeof(uint32_t));
	cudaMemcpyToSymbol(nc_d_height, &_height, sizeof(uint32_t));
}


static bool init[MAX_GPUS] = { 0 };

static bool ncDAGInit = false;

__host__
void nightcap_dag_update(int thr_id, int dev_id, uint32_t height, uint32_t throughput)
{
	uint32_t epoch = height / 400;
	NCGPUState* gpu_state = &nc_gpu_state[dev_id];
	NCThreadState* thread_state = &nc_thread_state[thr_id];

	applog(LOG_INFO, "nightcap_dag_update T%u gpu_state %x", thr_id, gpu_state);

	nightcap_dag_update_lock(dev_id);
	cudaSetDevice(dev_id);
	//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	//CUDA_LOG_ERROR();

	//applog(LOG_INFO, "nightcap_dag_update got dag lock");


	// Regen light cache if required
	uint32_t** cache_ptr = nightcap_lock_cache(epoch);
	uint32_t cache_size = nightcap_get_cache_size(height);
	if (!*cache_ptr || **cache_ptr != epoch)
	{
		// Regenerate light cache
		unsigned char seedhash[32];
		memset(seedhash, '\0', sizeof(seedhash));
		sph_blake256_context ctx_blake;
		for (uint32_t i = 0; i < epoch; i++) {
			sph_blake256_init(&ctx_blake);
			sph_blake256(&ctx_blake, seedhash, 32);
			sph_blake256_close(&ctx_blake, seedhash);
		}

		if (*cache_ptr)
			free(*cache_ptr);
		*cache_ptr = (uint32_t*)malloc(cache_size + sizeof(uint32_t));
		nightcap_generate_cache((uint32_t*)((*cache_ptr)+1), seedhash, cache_size);

		(*cache_ptr)[0] = epoch;
	}
	
	gpu_state->cache_size = cache_size;
	gpu_state->num_cache_nodes = gpu_state->cache_size / 32;

	//applog(LOG_INFO, "nightcap_dag_update got cache lock");

	// epoch we were working on isn't the current epoch?!
	if (thread_state->epoch != epoch)
	{
		//applog(LOG_INFO, "nightcap_dag_update epoch mismatch");
		// We'll need to update this thread state
	    init[thr_id] = false;
	    thread_state->epoch = epoch;

		// Regen if we are the first thread to init or set the current epoch
		if ((gpu_state->epoch != epoch && epoch > gpu_state->epoch) || !ncDAGInit)
		{
			uint32_t dag_size = nightcap_get_full_size(height);
			applog(LOG_INFO, "Regenerating dag, waiting for device %s to be ready...", device_name[dev_id]);
			ncDAGInit = true;

			// Force other threads to finish and block in this function
			restart_threads();
			cudaDeviceSynchronize();

			applog(LOG_INFO, "Device ready, regenerating dag for epoch %u...", epoch);

			gpu_state->dag_size = dag_size;
			gpu_state->num_dag_nodes = dag_size / 32;
			gpu_state->epoch = epoch;
			gpu_state->height = height;

			cudaDeviceReset();
			applog(LOG_INFO, "G%u device reset", dev_id);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&gpu_state->cache_nodes), gpu_state->cache_size));
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&gpu_state->dag_nodes), gpu_state->dag_size));

			// Copy over cache
			CUDA_SAFE_CALL(cudaMemcpy(gpu_state->cache_nodes, ((*cache_ptr)+1), gpu_state->cache_size, cudaMemcpyHostToDevice));

			cudaDeviceSynchronize();
			nightcap_unlock_cache(epoch); // free for other threads

			// Set node memory
			nightcap_set_mix_constants(gpu_state->dag_nodes, 
			                           gpu_state->num_dag_nodes,
			                           gpu_state->cache_nodes,
			                           gpu_state->num_cache_nodes,
			                           height);

			// Set blake stuff for dag gen
			blake256_cpu_init(thr_id, throughput);
			cuda_get_arch(thr_id);

			// Regen dag
			nightcap_recalc_dag(gpu_state->num_dag_nodes, 1UL << 21);
			applog(LOG_INFO, "T%u Dag regeneration complete", thr_id);

#ifdef NIGHTCAP_DEBUG_DAG
			// dump dag
			{
				uint32_t* dag_tmp = (uint32_t*)malloc(gpu_state->dag_size);
				CUDA_SAFE_CALL(cudaMemcpy(dag_tmp, gpu_state->dag_nodes, gpu_state->dag_size, cudaMemcpyDeviceToHost));

				FILE* fp = fopen("dag.dat", "wb");
				fwrite(dag_tmp, 1, gpu_state->dag_size, fp);
				fclose(fp);
				free(dag_tmp);
			}
#endif

			nightcap_dag_update_unlock(dev_id);

			applog(LOG_INFO, "T%u nightcap_dag_update returned", thr_id);
			return;
		}
	}
	

	// If we aren't init'd, init our cloverhash values
	if (!init[thr_id] || gpu_state->height != height)
	{
		applog(LOG_INFO, "T%u Using pregen dag with height %u.", thr_id, height);
		
		gpu_state->height = height;

		// Set node memory
		nightcap_set_mix_constants(gpu_state->dag_nodes, 
		                           gpu_state->num_dag_nodes,
		                           gpu_state->cache_nodes,
		                           gpu_state->num_cache_nodes,
		                           gpu_state->height);
	}
	nightcap_unlock_cache(epoch); // free for other threads

	nightcap_dag_update_unlock(dev_id);
	applog(LOG_INFO, "T%u nightcap_dag_update return NOUPDATE", thr_id);
}

//#define CLOVERHASH_COMPARE_ON_CPU


extern "C" int scanhash_nightcap(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] < 500) ? 18 : is_windows() ? 19 : 20;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	// Mutexes for gpu states need to be init'd
	nightcap_ensure_setup();

	if (opt_benchmark)
		ptarget[7] = 0x000f;
		
	//work->height = 25335; // HACK DEBUG

	// Make sure dag is up to date!
	nightcap_dag_update(thr_id, dev_id, work->height, throughput);

	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id])
	{
		applog(LOG_DEBUG, "reinit thread %u", thr_id);
		// NOTE: nightcap_dag_update resets device

		size_t matrix_sz = 16 * sizeof(uint64_t) * 4 * 3;
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		blake256_cpu_init(thr_id, throughput);
		skein256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);

		cuda_get_arch(thr_id); // cuda_arch[] also used in cubehash256

		// SM 3 implentation requires a bit more memory
		if (device_sm[dev_id] < 500 || cuda_arch[dev_id] < 500)
			matrix_sz = 16 * sizeof(uint64_t) * 4 * 4;
			
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2v2_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		api_set_throughput(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	//memset(pdata, '\0', 80); // DEBUG
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);


	blake256_cpu_setBlock_80(pdata);
	bmw256_setTarget(ptarget);

	#ifdef CLOVERHASH_COMPARE_ON_CPU
	uint64_t* temp_hashes = (uint64_t*)malloc((size_t)32 * throughput);
	#endif

	do {
		int order = 0;
		memset(work->nonces, 0, sizeof(work->nonces));

		// DEBUG uint64_t* temp_hashes = (uint64_t*)malloc((size_t)32 * throughput);

		// first pass
		blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput,pdata[19], d_hash[thr_id], order++);
		bmw256_cpu_hash_32_to_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		// mix
		nightcap_blakeKeccak_hashimoto_cpu_hash_32(thr_id, throughput,pdata[19], d_hash[thr_id], order++);



		// DEBUG
		/*
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(temp_hashes, d_hash[thr_id], (size_t)32 * throughput, cudaMemcpyDeviceToHost));

		for (uint32_t i=0; i<throughput; i++)
		{
			uint64_t hash[4];
			hash[0] = temp_hashes[i];
			hash[1] = temp_hashes[i + 1 * throughput];
			hash[2] = temp_hashes[i + 2 * throughput];
			hash[3] = temp_hashes[i + 3 * throughput];

			uint32_t* hash32 = (uint32_t*)hash;
			printf("Hashmimoto[%i] == %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n", i, hash32[0], hash32[1], hash32[2], hash32[3],hash32[4], hash32[5], hash32[6], hash32[7]);
		}
		exit(1);
		*/
		// END DEBUG


		// second pass
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput,pdata[19], d_hash[thr_id], order++);


		#ifdef CLOVERHASH_COMPARE_ON_CPU
		bmw256_cpu_hash_32_to_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		CUDA_SAFE_CALL(cudaMemcpy(temp_hashes, d_hash[thr_id], (size_t)32 * throughput, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		work->nonces[0] = 0;
		work->nonces[1] = 0;

		for (uint32_t i=0; i<throughput; i++)
		{
			uint64_t hash[4];
			hash[0] = temp_hashes[i];
			hash[1] = temp_hashes[i + 1 * throughput];
			hash[2] = temp_hashes[i + 2 * throughput];
			hash[3] = temp_hashes[i + 3 * throughput];
			uint32_t* vhash = (uint32_t*)&hash[0];
			const uint32_t Htarg = ptarget[7];
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				work->nonces[0] = pdata[19] + i;
				applog(LOG_INFO, "T%u found nonce %u on cpu in slot %i", thr_id, work->nonces[0], i);
				break;
			}
		}

		#else
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], work->nonces);
		#endif

		/* DEBUG OUT
		bmw256_cpu_hash_32_to_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		CUDA_SAFE_CALL(cudaMemcpy(temp_hashes, d_hash[thr_id], (size_t)32 * throughput, cudaMemcpyDeviceToHost));

		FILE* fp = fopen("out_hashes.dat", "wb");
		fwrite(temp_hashes, 1, (size_t)32 * throughput, fp);
		fclose(fp);

		for (uint32_t i=0; i<throughput; i++)
		{
			uint64_t hash[4];
			hash[0] = temp_hashes[i];
			hash[1] = temp_hashes[i + 1 * throughput];
			hash[2] = temp_hashes[i + 2 * throughput];
			hash[3] = temp_hashes[i + 3 * throughput];

			uint32_t* hash32 = (uint32_t*)hash;
			printf("Hashmimoto[%i] == %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n", i, hash32[0], hash32[1], hash32[2], hash32[3],hash32[4], hash32[5], hash32[6], hash32[7]);
		}

		free(temp_hashes);

		printf("Debug done\n");
		exit(1);
		*/



		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);

			nightcap_hash(vhash, endiandata, work->nonces[0], work->height);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					nightcap_hash(vhash, endiandata, work->nonces[1], work->height);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}

				#ifdef CLOVERHASH_COMPARE_ON_CPU
				free(temp_hashes);
				#endif
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU (%u > %u)!", work->nonces[0], vhash[7], Htarg);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && !abort_flag);

	#ifdef CLOVERHASH_COMPARE_ON_CPU
	free(temp_hashes);
	#endif

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}
