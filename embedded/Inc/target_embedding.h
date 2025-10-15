#ifndef TARGET_EMBEDDING_H
#define TARGET_EMBEDDING_H

#include "arm_math.h"
#include "app_config.h"
#include <stdint.h>

/* Target embedding constants */
#define EMBEDDING_SIZE 128
#define EMBEDDING_BANK_SIZE 10

/* Flash storage definitions */
#define EMBEDDING_FLASH_SECTOR_ADDR    0x73000000  /* Flash address for embedding storage */
#define EMBEDDING_FLASH_MAGIC          0xFACE1234  /* Magic number to verify valid data */
#define EMBEDDING_FLASH_BLOCK_SIZE     0x10000     /* 64KB block size for MX66UW1G45G */

typedef struct {
    uint32_t magic;                                    /* Magic number for validation */
    uint32_t version;                                  /* Version for future compatibility */
    int bank_count;                                    /* Number of stored embeddings */
    float embedding_bank[EMBEDDING_BANK_SIZE][EMBEDDING_SIZE]; /* Stored embeddings */
    uint32_t checksum;                                 /* Data integrity checksum */
} embedding_flash_data_t;

/* Global target embedding */
extern float target_embedding[EMBEDDING_SIZE];

/* Target embedding function prototypes */
void embeddings_bank_init(void);
int  embeddings_bank_add(const float *embedding);
void embeddings_bank_reset(void);
int  embeddings_bank_count(void);

/* Flash persistence functions */
int embeddings_save_to_flash(void);
int embeddings_load_from_flash(void);
int embeddings_erase_from_flash(void);

/* New functions for flash persistence */
int embeddings_save_to_flash(void);
int embeddings_load_from_flash(void);
int embeddings_erase_from_flash(void);

#endif /* TARGET_EMBEDDING_H */
