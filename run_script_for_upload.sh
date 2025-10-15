#!/bin/bash

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el /home/kyuubi/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr  -w Binary/ai_fsbl.hex

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el /home/kyuubi/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr -w Binary/face_detection_data.hex

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el /home/kyuubi/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr -w Binary/face_recognition_data.hex

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el /home/kyuubi/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr -w Binary/Project_signed.hex

