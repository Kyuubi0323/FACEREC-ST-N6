/**
 ******************************************************************************
 * @file    target_embedding.c
 * @brief   Target embedding management functions
 ******************************************************************************
 */

#include "target_embedding.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "stm32n6570_discovery_xspi.h"
/* ========================================================================= */
/* GLOBAL VARIABLES                                                          */
/* ========================================================================= */

// Dummy normalized embedding vector for testing face recognition system

// Real face embedding extracted from your image
// Real face embedding extracted from your image
// Real face embedding extracted from your image
// Updated from STM32N6 live capture
// Real face embedding extracted from your image
// Real face embedding extracted from your image
float target_embedding[EMBEDDING_SIZE] = {
    0.040647f, 0.026755f, 0.153597f, -0.080411f, -0.014650f, -0.086452f, 0.025299f, -0.108582f,
    0.021493f, -0.045558f, 0.056120f, 0.066751f, -0.042644f, -0.047429f, 0.023803f, -0.108559f,
    0.051514f, 0.010936f, -0.077689f, 0.197779f, 0.103035f, 0.050605f, 0.002235f, -0.063007f,
    -0.095122f, -0.092939f, -0.048167f, -0.005068f, 0.057043f, 0.117831f, -0.009633f, 0.067538f,
    0.070628f, -0.124384f, 0.036531f, 0.058474f, 0.140705f, -0.185638f, -0.089395f, -0.084688f,
    -0.029892f, -0.075863f, -0.095238f, 0.005727f, 0.088060f, -0.167979f, 0.067374f, 0.028175f,
    0.071912f, -0.053629f, 0.064472f, 0.014508f, -0.067513f, 0.006008f, 0.006562f, -0.047345f,
    -0.006382f, -0.103628f, 0.195097f, -0.078798f, 0.177411f, 0.047934f, -0.075454f, -0.161923f,
    -0.236855f, 0.073012f, -0.114272f, 0.029208f, -0.018735f, -0.076661f, 0.131198f, -0.030272f,
    0.104312f, 0.091074f, -0.119426f, 0.046640f, 0.063307f, -0.000465f, -0.107668f, -0.025739f,
    -0.033456f, -0.120411f, 0.118353f, -0.081199f, 0.009383f, -0.144818f, 0.099459f, -0.019652f,
    0.041115f, 0.062539f, 0.050633f, -0.032019f, -0.070235f, -0.034686f, -0.064582f, 0.091935f,
    -0.023405f, 0.139901f, 0.040438f, -0.026230f, -0.110393f, 0.083547f, 0.112278f, -0.087967f,
    0.108430f, 0.067396f, 0.094711f, 0.003457f, -0.055283f, -0.110831f, -0.018616f, -0.241341f,
    0.123861f, 0.103164f, 0.069854f, -0.129183f, 0.136289f, -0.024777f, -0.051062f, -0.112996f,
    -0.007972f, -0.019134f, 0.067339f, 0.127876f, -0.073011f, 0.047995f, 0.052420f, -0.014350f
};

static float embedding_bank[EMBEDDING_BANK_SIZE][EMBEDDING_SIZE];     /**< Bank of stored embeddings */
static int bank_count = 0;                                            /**< Current number of embeddings in bank */

/* ========================================================================= */
/* IMPLEMENTATION FUNCTIONS                                                  */
/* ========================================================================= */

/**
 * @brief Compute the target embedding as average of all embeddings in bank
 * @note This function is called automatically when embeddings are added
 */
/**
 * @brief Calculate simple checksum for data integrity
 */
static uint32_t calculate_checksum(const embedding_flash_data_t *data)
{
    uint32_t checksum = 0;
    const uint8_t *ptr = (const uint8_t *)data;
    size_t size = sizeof(embedding_flash_data_t) - sizeof(uint32_t); // Exclude checksum field
    
    for (size_t i = 0; i < size; i++) {
        checksum += ptr[i];
    }
    return checksum;
}


static void compute_target(void)
{
    if (bank_count == 0)
    {
        memset(target_embedding, 0, sizeof(target_embedding));
        return;
    }
    float sum[EMBEDDING_SIZE];
    memset(sum, 0, sizeof(sum));
    for (int n = 0; n < bank_count; n++)
    {
        for (int i = 0; i < EMBEDDING_SIZE; i++)
        {
            sum[i] += embedding_bank[n][i];
        }
    }
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        target_embedding[i] = sum[i] / (float)bank_count;
    }
    float norm = 0.f;
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        norm += target_embedding[i] * target_embedding[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.f)
    {
        for (int i = 0; i < EMBEDDING_SIZE; i++)
        {
            target_embedding[i] /= norm;
        }
    }
}

void embeddings_bank_init(void)
{
    bank_count = 0;
    memset(embedding_bank, 0, sizeof(embedding_bank));
    
    // LED: Flash red 3 times to indicate init started
    for (int i = 0; i < 3; i++) {
        BSP_LED_On(LED1);
        HAL_Delay(100);
        BSP_LED_Off(LED1);
        HAL_Delay(100);
    }
    
    // XSPI should already be initialized by main program
    // Just try to load from flash
    if (embeddings_load_from_flash() == 0) {
        // Successfully loaded from flash
        // LED: Flash green 5 times to indicate success
        for (int i = 0; i < 5; i++) {
            BSP_LED_On(LED2);
            HAL_Delay(200);
            BSP_LED_Off(LED2);
            HAL_Delay(200);
        }
    } else {
        // No data in flash or read failed
        // LED: Flash both LEDs alternating 3 times
        for (int i = 0; i < 3; i++) {
            BSP_LED_On(LED1);
            HAL_Delay(300);
            BSP_LED_Off(LED1);
            BSP_LED_On(LED2);
            HAL_Delay(300);
            BSP_LED_Off(LED2);
        }
        // Keep the hard-coded target_embedding for initial testing
    }
}

int embeddings_bank_add(const float *embedding)
{
    if (bank_count >= EMBEDDING_BANK_SIZE)
        return -1;
    float norm = 0.f;
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    if (norm == 0.f)
        return -1;
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        embedding_bank[bank_count][i] = embedding[i] / norm;
    }
    bank_count++;
    compute_target();
    
    // Auto-save to flash when bank is full (10 embeddings)
    if (bank_count == EMBEDDING_BANK_SIZE) {
        printf("Bank full (%d embeddings), saving to flash...\r\n", bank_count);
        
        // LED: Flash green rapidly 5 times to indicate saving
        for (int i = 0; i < 5; i++) {
            BSP_LED_On(LED2);
            HAL_Delay(50);
            BSP_LED_Off(LED2);
            HAL_Delay(50);
        }
        
        if (embeddings_save_to_flash() == 0) {
            printf("✓ Face training complete! Embeddings saved to flash.\r\n");
            // LED: Flash green slowly 3 times to confirm save success
            for (int i = 0; i < 3; i++) {
                BSP_LED_On(LED2);
                HAL_Delay(300);
                BSP_LED_Off(LED2);
                HAL_Delay(300);
            }
        } else {
            printf("✗ Failed to save embeddings to flash!\r\n");
            // LED: Flash red rapidly 10 times to indicate error
            for (int i = 0; i < 10; i++) {
                BSP_LED_On(LED1);
                HAL_Delay(50);
                BSP_LED_Off(LED1);
                HAL_Delay(50);
            }
        }
    }
    
    return bank_count;
}

void embeddings_bank_reset(void)
{
    // ONLY reset RAM, don't call embeddings_bank_init() 
    // and don't erase flash!
    bank_count = 0;
    memset(embedding_bank, 0, sizeof(embedding_bank));
    
    // Optionally reload from flash
    embeddings_load_from_flash();
}

int embeddings_bank_count(void)
{
    return bank_count;
}

/* ========================================================================= */
/* FLASH PERSISTENCE FUNCTIONS                                              */
/* ========================================================================= */

/**
 * @brief Calculate simple checksum for data integrity
 */


/**
 * @brief Save embedding bank to flash memory using STM32N6 XSPI
 * @return 0 on success, -1 on error
 */
int embeddings_save_to_flash(void)
{
    if (bank_count == 0) {
     
        return -1;
    }
    
    embedding_flash_data_t flash_data;
    
    // Prepare data structure
    flash_data.magic = EMBEDDING_FLASH_MAGIC;
    flash_data.version = 1;
    flash_data.bank_count = bank_count;
    
    // Copy embedding bank
    memcpy(flash_data.embedding_bank, embedding_bank, sizeof(embedding_bank));
    
    // Calculate checksum
    flash_data.checksum = calculate_checksum(&flash_data);
    
    // Erase flash block first
    
    if (embeddings_erase_from_flash() != 0) {
       
        return -1;
    }
    
    // Write to flash using STM32N6 XSPI BSP

    
    int32_t status = BSP_XSPI_NOR_Write(0, (const uint8_t *)&flash_data, 
                                        EMBEDDING_FLASH_SECTOR_ADDR, 
                                        sizeof(embedding_flash_data_t));
    
    if (status != BSP_ERROR_NONE) {
        
        return -1;
    }
    
    
    return 0;
}

/**
 * @brief Load embedding bank from flash memory using STM32N6 XSPI
 * @return 0 on success, -1 on error
 */
int embeddings_load_from_flash(void)
{
    embedding_flash_data_t flash_data;
    
    // Read from flash using STM32N6 XSPI BSP
    int32_t status = BSP_XSPI_NOR_Read(0, (uint8_t *)&flash_data, 
                                       EMBEDDING_FLASH_SECTOR_ADDR, 
                                       sizeof(embedding_flash_data_t));
    
    if (status != BSP_ERROR_NONE) {
        printf("Flash read failed: %ld\r\n", status);
        return -1;
    }
    
    // Check magic number
    if (flash_data.magic != EMBEDDING_FLASH_MAGIC) {
        printf("No valid embedding data in flash (magic: 0x%08lX)\r\n", flash_data.magic);
        return -1;
    }
    
    // Check version compatibility
    if (flash_data.version != 1) {
        printf("Unsupported embedding data version: %lu\r\n", flash_data.version);
        return -1;
    }
    
    // Verify checksum
    uint32_t calculated_checksum = calculate_checksum(&flash_data);
    if (calculated_checksum != flash_data.checksum) {
        printf("Embedding data corrupted (checksum: expected 0x%08lX, got 0x%08lX)\r\n", 
               flash_data.checksum, calculated_checksum);
        return -1;
    }
    
    // Validate bank count
    if (flash_data.bank_count < 0 || flash_data.bank_count > EMBEDDING_BANK_SIZE) {
        printf("Invalid bank count in flash: %d\r\n", flash_data.bank_count);
        return -1;
    }
    
    // Load data
    bank_count = flash_data.bank_count;
    memcpy(embedding_bank, flash_data.embedding_bank, sizeof(embedding_bank));
    
    // Recompute target embedding
    compute_target();
    
    printf("✓ Loaded %d embeddings from flash\r\n", bank_count);
    return 0;
}

/**
 * @brief Erase embedding data from flash using STM32N6 XSPI
 * @return 0 on success, -1 on error
 */
int embeddings_erase_from_flash(void)
{
    // Erase 64KB block using STM32N6 XSPI BSP
    int32_t status = BSP_XSPI_NOR_Erase_Block(0, EMBEDDING_FLASH_SECTOR_ADDR, 
                                               BSP_XSPI_NOR_ERASE_64K);
    
    if (status != BSP_ERROR_NONE) {
        return -1;
    }
    
    return 0;
}

/* ========================================================================= */
/* FLASH PERSISTENCE FUNCTIONS                                              */
/* ========================================================================= */





/**
 * @brief Load embedding bank from flash memory using STM32N6 XSPI
 * @return 0 on success, -1 on error
 */

