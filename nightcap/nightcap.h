#ifndef _NIGHTCAP_H
#define _NIGHTCAP_H

#include <stdint.h>
#include <math.h>


#define NIGHTCAP_WORD_BYTES 4
#define NIGHTCAP_DATASET_BYTES_INIT 536870912
#define NIGHTCAP_DATASET_BYTES_GROWTH 12582912
#define NIGHTCAP_CACHE_BYTES_INIT 8388608
#define NIGHTCAP_CACHE_BYTES_GROWTH 196608
#define NIGHTCAP_EPOCH_LENGTH 400
#define NIGHTCAP_CACHE_MULTIPLIER 64
#define NIGHTCAP_MIX_BYTES 64
#define NIGHTCAP_HASH_BYTES 32
#define NIGHTCAP_DATASET_PARENTS 256
#define NIGHTCAP_CACHE_ROUNDS 3
#define NIGHTCAP_ACCESSES 64
#define NIGHTCAP_FNV_PRIME 0x01000193

static int nightcap_is_prime(uint64_t number) {
    if (number <= 1) return 0;
    if((number % 2 == 0) && number > 2) return 0;
    for(uint64_t i = 3; i < sqrt(number); i += 2) {
        if(number % i == 0)
            return 0;
    }
    return 1;
}

static uint64_t nightcap_get_cache_size(uint64_t block_number) {
    uint64_t sz = NIGHTCAP_CACHE_BYTES_INIT + (NIGHTCAP_CACHE_BYTES_GROWTH * round(sqrt(6*(block_number / NIGHTCAP_EPOCH_LENGTH))));
    sz -= NIGHTCAP_HASH_BYTES;
    while (!nightcap_is_prime(sz / NIGHTCAP_HASH_BYTES)) {
        sz -= 2 * NIGHTCAP_HASH_BYTES;
    }
    return sz;
}

static uint64_t nightcap_get_full_size(uint64_t block_number) {
    uint64_t sz = NIGHTCAP_DATASET_BYTES_INIT + (NIGHTCAP_DATASET_BYTES_GROWTH * round(sqrt(6 * (block_number / NIGHTCAP_EPOCH_LENGTH))));
    sz -= NIGHTCAP_MIX_BYTES;
    while (!nightcap_is_prime(sz / NIGHTCAP_MIX_BYTES)) {
        sz -= 2 * NIGHTCAP_MIX_BYTES;
    }
    return sz;
}

void nightcap_generate_cache(uint32_t *cache_nodes_in, uint8_t * const seedhash, uint64_t cache_size);

uint32_t** nightcap_lock_cache(uint32_t epoch);
void nightcap_unlock_cache(uint32_t epoch);

typedef union _NightcapNode
{
	uint8_t bytes[8 * 4];
	uint32_t words[8];
	uint64_t double_words[8 / 2];
} NightcapNode;


void nightcap_hash(uint32_t* hash, uint32_t* endiandata, uint32_t nonce, uint32_t height);

bool nightcap_dag_update_lock(uint32_t dev_id);
void nightcap_dag_update_unlock(uint32_t dev_id);
void nightcap_ensure_setup();

#endif		// __ETHASH_H
