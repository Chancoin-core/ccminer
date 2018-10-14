#include "miner.h"
#include "algos.h"

#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
#include "nightcap/nightcap.h"
#include <assert.h>
#include <semaphore.h>


#ifdef _MSC_VER
#define restrict __restrict
#endif

static pthread_mutex_t NightcapCacheLock[2];
uint32_t* NightcapCache[2];

static pthread_mutex_t NightcapDagLock[MAX_GPUS];
static pthread_mutex_t NightcapSetupMutex = PTHREAD_MUTEX_INITIALIZER;

static bool NightcapDagLockInit = false;


#define WORD_BYTES 4
#define DATASET_BYTES_INIT 536870912
#define DATASET_BYTES_GROWTH 12582912
#define CACHE_BYTES_INIT 8388608
#define CACHE_BYTES_GROWTH 196608
#define EPOCH_LENGTH 400
#define CACHE_MULTIPLIER 64
#define MIX_BYTES 64
#define HASH_BYTES 32
#define DATASET_PARENTS 256
#define CACHE_ROUNDS 3
#define ACCESSES 64
#define FNV_PRIME 0x01000193


static uint32_t fnv(uint32_t v1, uint32_t v2) {
	return ((v1 * FNV_PRIME) ^ v2) % (0xffffffff);
}

struct CHashimotoResult {
	uint32_t cmix[4];
	uint32_t result[8];
};

bool nightcap_dag_update_lock(uint32_t dev_id)
{
	pthread_mutex_lock(&NightcapDagLock[dev_id]);
}

void nightcap_dag_update_unlock(uint32_t dev_id)
{
	pthread_mutex_unlock(&NightcapDagLock[dev_id]);
}

void nightcap_ensure_setup()
{
	pthread_mutex_lock(&NightcapSetupMutex);
	if (!NightcapDagLockInit)
	{
		if (opt_n_threads > cuda_num_devices())
		{
			applog(LOG_ERR, "nightcap: Multiple threads per GPU not permitted.");
			exit(1);
		}

		for (int i=0; i<MAX_GPUS; i++)
		{
			pthread_mutex_init(&NightcapDagLock[i], NULL);	
		}
		

		pthread_mutex_init(&NightcapCacheLock[0], NULL);	
		pthread_mutex_init(&NightcapCacheLock[1], NULL);
		NightcapDagLockInit = true;
	}
	pthread_mutex_unlock(&NightcapSetupMutex);

}

uint32_t** nightcap_lock_cache(uint32_t epoch)
{
	uint32_t idx = epoch % 2;

	pthread_mutex_lock(&NightcapCacheLock[idx]);

	return &NightcapCache[idx];
}

void nightcap_unlock_cache(uint32_t epoch)
{
	uint32_t idx = epoch % 2;

	pthread_mutex_unlock(&NightcapCacheLock[idx]);
}

static void lyra2re2_hash(const void* input, void* state, int length)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_cubehash256_context  ctx_cubehash;
	sph_skein256_context     ctx_skein;
	sph_bmw256_context       ctx_bmw;

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, length);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cubehash);
	sph_cubehash256(&ctx_cubehash, hashB, 32);
	sph_cubehash256_close(&ctx_cubehash, hashA);

	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cubehash);
	sph_cubehash256(&ctx_cubehash, hashA, 32);
	sph_cubehash256_close(&ctx_cubehash, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

static struct CHashimotoResult hashimoto(const uint8_t *blockToHash, const uint32_t *dag, uint64_t full_size, uint32_t height)
{
	uint64_t n = full_size / HASH_BYTES;
	uint64_t mixhashes = MIX_BYTES / HASH_BYTES;
	uint64_t wordhashes = MIX_BYTES / WORD_BYTES;
	uint8_t header[80];
	uint32_t hashedHeader[8];
	memcpy(header, blockToHash, 80);
	lyra2re2_hash((char *)blockToHash, (char*)hashedHeader, 80);
	uint32_t mix[MIX_BYTES / sizeof(uint32_t)];
	for (int i = 0; i < (MIX_BYTES / HASH_BYTES); i++) {
		memcpy(mix + (i * (HASH_BYTES / sizeof(uint32_t))), hashedHeader, HASH_BYTES);
	}
	for (int i = 0; i < ACCESSES; i++) {
		uint32_t p = fnv(i ^ hashedHeader[0], mix[i % (MIX_BYTES / sizeof(uint32_t))]) % (n / mixhashes) * mixhashes;
		uint32_t newdata[MIX_BYTES / sizeof(uint32_t)];
		for (int j = 0; j < mixhashes; j++) {
			uint64_t pj = (p + j) * 8;
			const uint32_t* item = dag + pj;
			memcpy(newdata + (j * 8), item, HASH_BYTES);
		}
		for (int i = 0; i < MIX_BYTES / sizeof(uint32_t); i++) {
			mix[i] = fnv(mix[i], newdata[i]);
		}
	}
	uint32_t cmix[4];
	for (int i = 0; i < MIX_BYTES / sizeof(uint32_t); i += 4) {
		cmix[i / 4] = fnv(fnv(fnv(mix[i], mix[i + 1]), mix[i + 2]), mix[i + 3]);
	}
	struct CHashimotoResult result;
	memcpy(result.cmix, cmix, MIX_BYTES / 4);
	uint8_t hash[52];
	memcpy(hash, hashedHeader, 32);
	memcpy(hash + 36, cmix, 16);
	memcpy(hash + 32, &height, 4);
	lyra2re2_hash((char *)hash, (char *)result.result, 52);
	return result;

}

// Calc node on fly
void my_calc_dataset_item(const uint32_t *cache, uint64_t i, uint64_t cache_size, uint32_t *out_mix)
{
	sph_blake256_context ctx;
	uint64_t items = cache_size / HASH_BYTES;
	uint64_t hashwords = HASH_BYTES / WORD_BYTES;
	memcpy(out_mix, cache + ((i % items) * (HASH_BYTES / sizeof(uint32_t))), HASH_BYTES);
	out_mix[0] ^= i;

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, out_mix, HASH_BYTES);
	sph_blake256_close(&ctx, out_mix);

       assert(cache != NULL);

	for (uint64_t parent = 0; parent < DATASET_PARENTS; parent++) {
		uint64_t index = fnv(i ^ parent, out_mix[parent % (HASH_BYTES / sizeof(uint32_t))]) % items;
		for (uint64_t dword = 0; dword < (HASH_BYTES / sizeof(uint32_t)); dword++) {
			out_mix[dword] = fnv(out_mix[dword], cache[index * (HASH_BYTES / sizeof(uint32_t))]);
		}
	}

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, out_mix, HASH_BYTES);
	sph_blake256_close(&ctx, out_mix);
}

// Same as hashimoto except it doesn't use the dag
static struct CHashimotoResult light_hashimoto(const uint8_t *blockToHash, const uint32_t *cache, uint64_t cache_size, uint64_t full_size, uint32_t height)
{
	//assert(cache_size == get_cache_size(height));

	const uint64_t n = full_size / HASH_BYTES;
	const uint64_t mixhashes = MIX_BYTES / HASH_BYTES;
	const uint64_t wordhashes = MIX_BYTES / WORD_BYTES;
	uint32_t hashedHeader[8];
	uint32_t mix[MIX_BYTES / sizeof(uint32_t)];
	uint32_t cmix[4];
	lyra2re2_hash(blockToHash, (char*)hashedHeader, 80);
	for (uint64_t i = 0; i < mixhashes; i++) {
		memcpy(mix + (i * (HASH_BYTES / sizeof(uint32_t))), hashedHeader, HASH_BYTES);
	}
	for (uint64_t i = 0; i < ACCESSES; i++) {
		uint32_t target = fnv(i ^ hashedHeader[0], mix[i % (MIX_BYTES / sizeof(uint32_t))]) % (n / mixhashes) * mixhashes;
		uint32_t mapdata[MIX_BYTES / sizeof(uint32_t)];
		for (uint64_t mixhash = 0; mixhash < mixhashes; mixhash++) {
			//CDAGNode node = GetNode(target + mixhash, header.height);
			//assert((mixhash * (HASH_BYTES / sizeof(uint32_t))) < 16);

			uint32_t node[HASH_BYTES / sizeof(uint32_t)];
			my_calc_dataset_item(cache, target + mixhash, cache_size, node);
			memcpy(mapdata + (mixhash * (HASH_BYTES / sizeof(uint32_t))), node, HASH_BYTES);
		}
		for (uint64_t dword = 0; dword < (MIX_BYTES / sizeof(uint32_t)); dword++) {
			mix[dword] = fnv(mix[dword], mapdata[dword]);
			//assert(dword < sizeof(mix));
		}
	}
	for (uint64_t i = 0; i < MIX_BYTES / sizeof(uint32_t); i += sizeof(uint32_t)) {
		cmix[i / sizeof(uint32_t)] = fnv(fnv(fnv(mix[i], mix[i + 1]), mix[i + 2]), mix[i + 3]);
	}

	struct CHashimotoResult result;
	memcpy(result.cmix, cmix, MIX_BYTES / 4);
	uint8_t hash[52];
	memcpy(hash, hashedHeader, 32);
	memcpy(hash + 32, &height, 4);
	memcpy(hash + 36, cmix, 16);
	lyra2re2_hash((char *)hash, (char *)result.result, 52);
	return result;
}

// Output (cache_nodes) MUST have at least cache_size bytes
void nightcap_generate_cache(uint32_t *cache, uint8_t* const seed, uint64_t cache_size)
{
	uint64_t items = cache_size / NIGHTCAP_HASH_BYTES;
	sph_blake256_context ctx_blake;
	int64_t hashwords = NIGHTCAP_HASH_BYTES / NIGHTCAP_WORD_BYTES;

	sph_blake256_context ctx;
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, seed, NIGHTCAP_HASH_BYTES);
	sph_blake256_close(&ctx, cache);

	for (uint64_t i = 1; i < items; i++) {
		sph_blake256_init(&ctx);
		sph_blake256(&ctx, cache + ((i - 1) * (hashwords)), NIGHTCAP_HASH_BYTES);
		sph_blake256_close(&ctx, cache + i*hashwords);
	}
	for (uint64_t round = 0; round < NIGHTCAP_CACHE_ROUNDS; round++) {
		//3 round randmemohash.
		for (uint64_t i = 0; i < items; i++) {
			uint64_t target = cache[(i * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t)))] % items;
			uint64_t mapper = (i - 1 + items) % items;
			/* Map target onto mapper, hash it,
			* then replace the current cache item with the 32 byte result. */
			uint32_t item[NIGHTCAP_HASH_BYTES / sizeof(uint32_t)];
			for (uint64_t dword = 0; dword < (NIGHTCAP_HASH_BYTES / sizeof(uint32_t)); dword++) {
				item[dword] = cache[(mapper * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))) + dword]
				            ^ cache[(target * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))) + dword];
			}
			sph_blake256_init(&ctx);
			sph_blake256(&ctx, item, NIGHTCAP_HASH_BYTES);
			sph_blake256_close(&ctx, item);
			memcpy(cache + (i * (NIGHTCAP_HASH_BYTES / sizeof(uint32_t))), item, NIGHTCAP_HASH_BYTES);
		}
	}
}


void test_hashimoto(uint32_t height)
{
	uint32_t epoch = (height / 400);
	uint32_t idx = epoch % 2;
	uint32_t endiandata[20];
	uint64_t cache_size = nightcap_get_cache_size(height);
	uint64_t full_size = nightcap_get_full_size(height);

	memset(endiandata, '\0', sizeof(endiandata));

	pthread_mutex_lock(&NightcapCacheLock[idx]);

	struct CHashimotoResult res = light_hashimoto((uint8_t*)endiandata, (uint32_t*)(NightcapCache[idx] + 1), cache_size, full_size, height);

	printf("LightHashimoto(%u, %u) -> %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n", full_size, height, res.result[0], res.result[1], res.result[2], res.result[3], res.result[4], res.result[5], res.result[6], res.result[7]);

	pthread_mutex_unlock(&NightcapCacheLock[idx]);
}

void nightcap_hash(uint32_t* hash, uint32_t* endiandata, uint32_t nonce, uint32_t height)
{
	uint32_t epoch = (height / 400);
	uint32_t idx = epoch % 2;
	uint64_t cache_size = nightcap_get_cache_size(height);
	uint64_t full_size = nightcap_get_full_size(height);

	applog(LOG_DEBUG, "nightcap_regenhash: nonce check %u for height %u.", nonce, height);

        //printf("Cache hash epoch %u idx %u\n", epoch, idx);

	pthread_mutex_lock(&NightcapCacheLock[idx]);

        assert(NightcapCache[idx] != NULL);

	struct CHashimotoResult res = light_hashimoto((uint8_t*)endiandata, (uint32_t*)(NightcapCache[idx] + 1), cache_size, full_size, height);

	pthread_mutex_unlock(&NightcapCacheLock[idx]);

	memcpy(hash, res.result, 32);

	char *DbgHash = bin2hex(hash, 32);
	applog(LOG_DEBUG, "Regenhash result: %s.", DbgHash);
	free(DbgHash);
}

