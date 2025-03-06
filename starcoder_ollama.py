#!/usr/bin/env python3
"""
Script to read requirements from llvm-exec/source-code-data/llvm/llvm-exec/requirements,
generate C programs using Ollama's StarCoder model, and save the outputs to a specified directory.

Usage:
    python ollama_starcoder.py --requirements-dir=/path/to/llvm-exec/source-code-data/llvm/llvm-exec/requirements --output-dir=/path/to/output --num=10

The script will:
1. Scan the requirements directory for .txt files
2. Send each requirement to the Ollama StarCoder model
3. Extract C programs from the responses
4. Save the extracted C programs directly to the output directory with passname_timestamp naming
"""

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
import requests


class Logger:
    def __init__(self, log_file: Path, is_print=True) -> None:
        self.log_file = log_file
        self.is_print = is_print

        # Initialize log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        with open(self.log_file, "a") as f:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            f.write("====================\n")
            f.write(f"[{formatted_datetime}] Start logging.\n")

    def log(self, msg):
        if self.is_print:
            print(msg)
        timestamp = datetime.now().strftime("%d.%b %Y %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")


class OllamaStarCoder:
    def __init__(self, api_url="http://localhost:11434/api/generate", temperature=0.7) -> None:
        self.api_url = api_url
        self.temperature = temperature
        self.model = "starcoder"  # Assuming the model is loaded in Ollama

    def generate(self, prompt, num_samples=1, max_retries=3):
        """Generate code using Ollama's StarCoder model with retries"""
        outputs = []
        
        # Extract requirement name to tailor the prompt
        requirement_name = prompt.split('\n')[0] if '\n' in prompt else prompt[:50]
        
        for i in range(num_samples):
            retries = 0
            success = False
            
            # Analyze the prompt to determine if it's LLVM-specific
            is_llvm = any(keyword in prompt.lower() for keyword in ['llvm', 'pass', 'optimization', 'compiler'])
            is_cpp = any(keyword in prompt.lower() for keyword in ['c++', 'cpp', 'class', 'stl', 'template'])
            
            # Customize prompt based on content
            if is_llvm and is_cpp:
                # LLVM C++ specific prompt
                enhanced_prompt = f"""
Generate a complete, standalone C program that demonstrates the functionality similar to what an LLVM {requirement_name} would optimize.
The program should contain patterns that this pass would typically handle.

Example: If this is a dead code elimination pass, create C code with obvious dead code.
Do not include any C++ features, LLVM specific code, or explanations - just pure C code that compiles.

{prompt}

Return ONLY the C code within triple backticks. No explanations or comments outside the code block.
"""
            else:
                # Generic C program prompt
                enhanced_prompt = f"""
Generate a complete, standalone C program based on the following requirement:

{prompt}

Create a working, compilable C program (not C++) that demonstrates the functionality.
Return ONLY the C code within triple backticks. No explanations or comments outside the code block.
"""
            
            while not success and retries < max_retries:
                data = {
                    "model": self.model,
                    "prompt": enhanced_prompt,
                    "temperature": self.temperature + (retries * 0.1),  # Increase temperature slightly on retries
                    "stream": False,
                    "max_tokens": 2048
                }
                
                try:
                    response = requests.post(self.api_url, json=data)
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("response", "")
                        
                        # Check if the response seems valid
                        if response_text and len(response_text.strip()) > 0:
                            outputs.append(response_text)
                            success = True
                        else:
                            print(f"Empty response received on attempt {retries+1}. Retrying...")
                            retries += 1
                    else:
                        print(f"Error from Ollama API: {response.status_code} - {response.text}")
                        retries += 1
                except Exception as e:
                    print(f"Exception when calling Ollama API: {str(e)}")
                    retries += 1
                
                # Wait a bit before retrying with increasing backoff
                if not success and retries < max_retries:
                    time.sleep(2 * (retries + 1))
            
            # If all retries failed, add an empty response
            if not success:
                print(f"Failed to get valid response after {max_retries} attempts")
                outputs.append("")
                
        return outputs


def extract_c_program(response):
    """Extract C program from the response"""
    if not response or len(response.strip()) == 0:
        return None
        
    # First try to extract code between C code blocks
    c_code_pattern = re.compile(r"```c\n(.*?)```", re.DOTALL)
    matches = c_code_pattern.findall(response)
    
    if matches:
        code = matches[0].strip()
        if is_valid_c_program(code):
            return code
    
    # If no C code blocks found, check for generic code blocks
    code_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = code_pattern.findall(response)
    
    if matches:
        for match in matches:
            code = match.strip()
            if is_valid_c_program(code):
                return code
    
    # If no valid code found in blocks, try to extract code-like content
    potential_code_sections = []
    
    # Look for C-like syntax patterns that might indicate code outside of code blocks
    lines = response.split('\n')
    current_section = []
    in_code_section = False
    
    for line in lines:
        # Signs that might indicate start of code section
        if ('{' in line or line.strip().endswith(';') or 
            any(indicator in line for indicator in ["#include", "int main", "void main"])):
            in_code_section = True
        
        # If we're in what appears to be code, collect it
        if in_code_section:
            current_section.append(line)
            
            # Signs that might indicate end of code section
            if '}' in line and len(current_section) > 3:  # Must have reasonable length
                potential_code_sections.append('\n'.join(current_section))
                current_section = []
                in_code_section = False
    
    # If we were collecting a section but hit end of text
    if current_section:
        potential_code_sections.append('\n'.join(current_section))
    
    # Try each potential code section
    for section in potential_code_sections:
        if is_valid_c_program(section):
            return section
    
    # If all else fails, check if the entire response looks like valid C code
    full_text = response.strip()
    if is_valid_c_program(full_text):
        return full_text
        
    return None

def is_valid_c_program(code):
    """
    Use heuristics to determine if the text is likely a valid C program
    """
    if not code or len(code.strip()) < 10:  # Too short to be useful
        return False
    
    # Special case for LLVM pass demonstrations - lower our standards a bit
    # since we're dealing with compiler optimization examples
    if any(keyword in code.lower() for keyword in ['llvm', 'pass', 'optimization']):
        # For LLVM examples, just check for basic C structure
        return '{' in code and '}' in code and ';' in code
        
    # Look for common C program indicators
    c_indicators = [
        "#include", "int main", "void main", "return 0", "printf", "scanf",
        "malloc", "free", "struct", "typedef", "enum", "const", "static",
        "for(", "for (", "while(", "while (", "if(", "if (", "else{", "else {"
    ]
    
    # Check if the code contains at least one C indicator
    has_c_indicator = any(indicator in code for indicator in c_indicators)
    
    # Check if it contains curly braces (a basic feature of C programs)
    has_curly_braces = '{' in code and '}' in code
    
    # Check if it contains semicolons (another basic feature of C)
    has_semicolons = ';' in code
    
    # If it's at least structured like C code but doesn't have typical C program indicators,
    # look for function definitions which are common in code examples
    function_pattern = re.search(r'\w+\s+\w+\s*\([^)]*\)\s*{', code)
    has_function_def = function_pattern is not None
    
    # More lenient check for code fragments that might demonstrate compiler optimizations
    if has_curly_braces and has_semicolons and has_function_def:
        return True
    
    # Return true if it has enough C-like features
    return has_c_indicator and has_curly_braces and has_semicolons


def scan_requirements(requirements_dir: Path, existing_reqs: set):
    """Scan for new requirement files"""
    new_reqs = set()
    
    for req_file in requirements_dir.glob("**/*.txt"):
        if req_file.is_file() and req_file not in existing_reqs:
            new_reqs.add(req_file)
            
    return new_reqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama StarCoder for LLVM requirements")
    parser.add_argument("--requirements-dir", type=str, 
                        default="llvm-exec/source-code-data/llvm/llvm-exec/requirements",
                        help="Directory containing requirement files")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to store generated code")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of samples to generate per requirement")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--api-url", type=str, default="http://localhost:11434/api/generate",
                        help="Ollama API URL")
    parser.add_argument("--sleep-time", type=int, default=30,
                        help="Sleep time between scans (seconds)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run in continuous mode, checking for new requirements")
    
    args = parser.parse_args()
    
    # Set up directories
    requirements_dir = Path(args.requirements_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(output_dir / "log.txt")
    
    # Log arguments
    logger.log("Arguments for Ollama StarCoder service")
    for k, v in vars(args).items():
        logger.log(f"  {k}: {v}")
    
    # Initialize Ollama client
    ollama = OllamaStarCoder(api_url=args.api_url, temperature=args.temperature)
    
    # Track processed requirements
    existing_reqs = set()
    
    # Main loop
    while True:
        new_reqs = scan_requirements(requirements_dir, existing_reqs)
        
        if len(new_reqs) == 0:
            if not args.continuous:
                logger.log("No requirements found. Exiting.")
                break
                
            logger.log(f"No new requirements, sleeping for {args.sleep_time}s...")
            time.sleep(args.sleep_time)
            continue
        
        # Process new requirements
        length = len(new_reqs)
        logger.log(f"Found {length} new requirements, starting generation...")
        
        for idx, req_file in enumerate(new_reqs):
            existing_reqs.add(req_file)
            
            # Get the passname (requirement name)
            passname = req_file.stem
            
            logger.log(f"[{idx+1}/{length}] {passname}: generating")
            
            # Read requirement
            requirement = req_file.read_text(encoding="utf-8", errors="ignore")
            
            # Generate responses
            try:
                t_start = time.time()
                responses = ollama.generate(requirement, num_samples=args.num)
                generation_time = time.time() - t_start
                logger.log(f"[{idx+1}/{length}] {passname}: generated {len(responses)} responses in {generation_time:.2f}s")
                
                # Process each response
                for i, response in enumerate(responses):
                    # Extract C program
                    c_program = extract_c_program(response)
                    
                    # Only save if we extracted a valid C program
                    if c_program is not None:
                        # Generate timestamp
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
                        
                        # Save to file directly in output directory
                        output_file = output_dir / f"{passname}_{timestamp}.c"
                        output_file.write_text(c_program)
                        logger.log(f"Saved {output_file}")
                    else:
                        logger.log(f"Response {i+1} for {passname} did not contain a valid C program")
                    
            except Exception as e:
                logger.log(f"Error processing {passname}: {str(e)}")
        
        # Exit if not running in continuous mode
        if not args.continuous:
            logger.log("All requirements processed. Exiting.")
            break