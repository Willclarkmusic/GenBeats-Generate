# Claude Code Instructions

## Core Principles

### Critical Thinking & Code Quality

- **Question instructions**: Don't blindly follow requests. Analyze if the approach makes sense.
- **Suggest improvements**: If you see a better way to implement something, propose it.
- **Validate dependencies**: Check if all package versions are compatible and necessary.
- **Review architecture**: Ensure the overall design is sound before implementing.
- **Test assumptions**: Verify that the proposed solution actually solves the problem.

### Code Standards

- **Clean, readable code**: Write code that is self-documenting with clear variable names
- **Comprehensive comments**: Explain complex logic, not just what the code does
- **Error handling**: Include proper try/catch blocks and graceful error responses
- **Type hints**: Use Python type hints for better code clarity
- **Consistent formatting**: Follow PEP 8 standards consistently
- **Modular design**: Break code into logical, reusable functions and classes

### Development Environment

- **Use pip exclusively**: No conda, no other package managers
- **Virtual environment**: Always use `python -m venv venv` in project directory
- **Requirements management**: Maintain clear requirements.txt with specific versions
- **Windows compatibility**: Ensure all scripts work on Windows with proper path handling
- **GPU detection**: Implement proper CUDA availability checks with CPU fallback

## Project Requirements - AnimateDiff Backend

### Architecture Review

Before implementing, verify:

- Is FastAPI the right choice for this use case?
- Are the proposed dependencies actually needed?
- Is the password protection approach secure enough?
- Will the serverless deployment strategy actually save costs?
- Are there simpler alternatives to achieve the same goals?

### Implementation Standards

#### API Design

```python
# Good: Clear, typed, documented
@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    current_user: User = Depends(get_current_user)
) -> VideoGenerationResponse:
    """
    Generate a short video using AnimateDiff.

    Args:
        request: Video generation parameters
        current_user: Authenticated user

    Returns:
        Video generation job information

    Raises:
        HTTPException: If generation fails or invalid parameters
    """
    try:
        # Implementation here
        pass
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

# Bad: Unclear, untyped, no documentation
@app.post("/generate")
def gen(prompt, steps=28):
    # Generate video somehow
    pass
```

#### Dependency Management

```txt
# Good: Specific versions, clear purpose
torch==2.1.0+cu118  # PyTorch with CUDA 11.8 support
fastapi==0.104.0    # Web framework
diffusers==0.25.0   # Hugging Face diffusion models

# Bad: Unclear versions, unnecessary packages
torch>=2.0
some-random-package
```

#### Error Handling

```python
# Good: Comprehensive error handling
async def download_models():
    """Download required models with proper error handling."""
    try:
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)

        logger.info("Downloading AnimateDiff models...")
        # Download logic with progress tracking

    except ConnectionError as e:
        logger.error(f"Network error downloading models: {e}")
        raise RuntimeError("Failed to download models - check internet connection")
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise RuntimeError("Permission denied - check directory permissions")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Model download failed: {str(e)}")

# Bad: No error handling
def download_models():
    # Download stuff, hope it works
    pass
```

### Critical Review Checklist

Before implementing any feature, ask:

1. **Necessity**: Is this feature actually needed for the MVP?
2. **Complexity**: Is this the simplest solution that works?
3. **Performance**: Will this approach scale with multiple users?
4. **Security**: Are there any security vulnerabilities?
5. **Maintainability**: Can someone else understand and modify this code?
6. **Testing**: How will this be tested and validated?
7. **Documentation**: Is the purpose and usage clear?

### Challenge Common Requests

When the user asks for something, consider:

- **"Add this dependency"** → Is it really needed? What does it add? Are there simpler alternatives?
- **"Make it faster"** → Where is the actual bottleneck? Don't optimize prematurely.
- **"Add this feature"** → Does it fit the core use case? Will it complicate the codebase?
- **"Use this pattern"** → Is it appropriate for this project size and complexity?

### Specific AnimateDiff Considerations

#### Model Management

- Question: Do we need to download all possible models or just the essential ones?
- Verify: Are the model sizes reasonable for the target deployment environment?
- Consider: Should models be downloaded at build time or runtime?

#### GPU/CPU Handling

- Validate: Is the GPU detection logic robust across different environments?
- Consider: What happens when GPU memory is exhausted?
- Review: Is the CPU fallback actually usable for video generation?

#### API Security

- Challenge: Is simple password protection sufficient for personal use?
- Consider: Should there be rate limiting even for personal use?
- Verify: Are the JWT tokens properly secured?

#### Deployment Strategy

- Question: Is Docker the best approach for serverless deployment?
- Evaluate: Will the container size be practical for serverless platforms?
- Consider: Are there platform-specific optimizations we should implement?

## Development Workflow

### Before Starting Implementation

1. **Analyze the request**: Understand the real problem being solved
2. **Propose alternatives**: Suggest simpler or more robust approaches if applicable
3. **Validate architecture**: Ensure the overall design makes sense
4. **Check dependencies**: Verify all packages are necessary and compatible

### During Implementation

1. **Write tests first**: For critical functionality, implement tests before code
2. **Document as you go**: Add comments and docstrings immediately
3. **Handle errors gracefully**: Every external call should have error handling
4. **Log appropriately**: Add logging for debugging and monitoring

### After Implementation

1. **Review code quality**: Is it readable and maintainable?
2. **Test thoroughly**: Verify functionality in different scenarios
3. **Document usage**: Provide clear instructions for setup and deployment
4. **Validate security**: Check for potential vulnerabilities

## Communication Guidelines

### When Disagreeing with Instructions

- Explain your reasoning clearly
- Propose specific alternatives
- Highlight potential issues with the requested approach
- Ask clarifying questions about the requirements

### When Suggesting Improvements

- Explain the benefits of your suggestion
- Estimate the implementation effort
- Consider backwards compatibility
- Provide concrete examples

Remember: Your job is to create the best possible solution, not just to follow instructions blindly. Use your expertise to guide the project toward success.
