#!/usr/bin/env python3
"""
STM32 Development Environment Setup Script
Reads stm32_tools_config.json and exports all tool paths as environment variables
"""

import json
import os
import sys
from pathlib import Path

def load_config(config_path):
    """Load the STM32 tools configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def export_environment_variables(config):
    """Generate shell commands to export environment variables"""
    
    commands = []
    commands.append("# STM32 Development Tools Environment Variables")
    commands.append("# Generated from stm32_tools_config.json")
    commands.append("")
    
    # Export tool paths
    if 'tools' in config:
        for tool_name, tool_config in config['tools'].items():
            if 'path' in tool_config:
                path = tool_config['path']
                
                # Create environment variable name (uppercase, replace hyphens with underscores)
                env_var_name = f"STM32_{tool_name.upper().replace('-', '_')}_PATH"
                
                # Export the path
                commands.append(f'export {env_var_name}="{path}"')
                
                # If it's a directory path (ends with /), add to PATH
                if path.endswith('/') or os.path.isdir(path):
                    commands.append(f'export PATH="{path}:$PATH"')
                # If it's a file path, add its directory to PATH
                elif os.path.isfile(path):
                    dir_path = os.path.dirname(path)
                    commands.append(f'export PATH="{dir_path}:$PATH"')
    
    commands.append("")
    
    # Export memory layout information
    if 'memory_layout' in config:
        commands.append("# Memory Layout")
        for key, value in config['memory_layout'].items():
            env_var_name = f"STM32_MEMORY_{key.upper()}"
            commands.append(f'export {env_var_name}="{value}"')
        commands.append("")
    
    # Export build configuration
    if 'build' in config:
        commands.append("# Build Configuration")
        for key, value in config['build'].items():
            if key != 'description':
                env_var_name = f"STM32_BUILD_{key.upper()}"
                commands.append(f'export {env_var_name}="{value}"')
        commands.append("")
    
    # Export model addresses
    if 'models' in config:
        commands.append("# Neural Network Model Addresses")
        for model_name, model_config in config['models'].items():
            if 'address' in model_config:
                env_var_name = f"STM32_MODEL_{model_name.upper()}_ADDRESS"
                commands.append(f'export {env_var_name}="{model_config["address"]}"')
        commands.append("")
    
    # Export embedding storage configuration
    if 'embedding_storage' in config:
        commands.append("# Embedding Storage Configuration")
        for key, value in config['embedding_storage'].items():
            env_var_name = f"STM32_EMBEDDING_{key.upper()}"
            commands.append(f'export {env_var_name}="{value}"')
        commands.append("")
    
    # Add convenience aliases
    commands.append("# Convenience Aliases")
    commands.append('alias stm32prog="$STM32_STM32PROGRAMMER_PATH"')
    commands.append('alias stm32sign="$STM32_STM32SIGNINGTOOL_PATH"')
    commands.append('alias stedgeai="$STM32_STM32EDGEAI_PATH"')
    commands.append('alias stm32ide="$STM32_STM32CUBEIDE_PATH"')
    commands.append("")
    
    # Add verification function
    commands.append("# Verification Function")
    commands.append("verify_stm32_tools() {")
    commands.append('    echo "Verifying STM32 development tools..."')
    commands.append('    command -v $STM32_STM32PROGRAMMER_PATH >/dev/null 2>&1 && echo "✓ STM32 Programmer found" || echo "✗ STM32 Programmer not found"')
    commands.append('    command -v $STM32_STM32SIGNINGTOOL_PATH >/dev/null 2>&1 && echo "✓ STM32 Signing Tool found" || echo "✗ STM32 Signing Tool not found"')
    commands.append('    command -v $STM32_STM32EDGEAI_PATH >/dev/null 2>&1 && echo "✓ STM32 Edge AI found" || echo "✗ STM32 Edge AI not found"')
    commands.append('    command -v arm-none-eabi-gcc >/dev/null 2>&1 && echo "✓ ARM GCC Toolchain found" || echo "✗ ARM GCC Toolchain not found"')
    commands.append("}")
    commands.append("")
    
    return '\n'.join(commands)

def main():
    # Find the configuration file
    script_dir = Path(__file__)
    project_root = script_dir.parent
    config_file = project_root / 'stm32_tools_config.json'
    
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_file)
    
    # Generate shell commands
    shell_commands = export_environment_variables(config)
    
    # Output options
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(shell_commands)
        os.chmod(output_file, 0o755)
        print(f"Environment setup script saved to: {output_file}")
        print(f"Run: source {output_file}")
    else:
        # Print to stdout (can be sourced directly)
        print(shell_commands)

if __name__ == '__main__':
    main()
