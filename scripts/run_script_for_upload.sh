#!/bin/bash

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CUBE_PROGRAMMER_PATH="/home/kyuubi/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
EXTERNAL_LOADER_PATH="$CUBE_PROGRAMMER_PATH/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr"

echo "Project root: $PROJECT_ROOT"
echo "Using external loader: $EXTERNAL_LOADER_PATH"

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EXTERNAL_LOADER_PATH" -w "$PROJECT_ROOT/embedded/Binary/ai_fsbl.hex"

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EXTERNAL_LOADER_PATH" -w "$PROJECT_ROOT/embedded/Binary/face_detection_data.hex"

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EXTERNAL_LOADER_PATH" -w "$PROJECT_ROOT/embedded/Binary/face_recognition_data.hex"

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EXTERNAL_LOADER_PATH" -w "$PROJECT_ROOT/embedded/Binary/Project_signed.hex"

