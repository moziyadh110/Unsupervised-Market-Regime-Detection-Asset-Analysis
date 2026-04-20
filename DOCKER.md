# Docker Quick Start Guide

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Build and start Jupyter Lab
docker-compose up

# 2. Open browser to http://localhost:8888
# Jupyter Lab will be running (no password required)

# 3. Stop the container
docker-compose down
```

### Option 2: Using Dockerfile Only

```bash
# 1. Build the image
docker build -t geopolitical-risk-analysis .

# 2. Run Jupyter Lab
docker run -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/results:/app/results \
  geopolitical-risk-analysis

# 3. Open browser to http://localhost:8888
```

## Common Commands

### Run Jupyter Lab
```bash
docker-compose up
```

### Run Tests
```bash
docker-compose run tests
# Or with coverage report
docker-compose run tests pytest tests/ -v --cov=src --cov-report=html
```

### Run Python Shell
```bash
docker-compose run --rm python bash
# Then inside container:
python
>>> from src import models
>>> # Use the modules
```

### Run Specific Notebook
```bash
docker-compose run --rm jupyter jupyter nbconvert \
  --to notebook --execute notebooks/02_data_cleaning.ipynb
```

### Access Container Shell
```bash
docker-compose run --rm jupyter bash
```

## Volume Mounts

The following directories are mounted for persistence:
- `./data` → Container's `/app/data`
- `./notebooks` → Container's `/app/notebooks`
- `./results` → Container's `/app/results`
- `./models` → Container's `/app/models`

Changes in these directories persist after container stops.

## Environment Variables

You can customize the container with environment variables:

```bash
# In docker-compose.yml, add under 'environment':
environment:
  - JUPYTER_PORT=8888
  - PYTHONUNBUFFERED=1
```

## Building from Scratch

```bash
# Remove old containers and images
docker-compose down --rmi all

# Rebuild
docker-compose build --no-cache

# Start fresh
docker-compose up
```

## Troubleshooting

### Port 8888 already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8889:8888"  # Use 8889 on host
```

### Permission issues (Linux)
```bash
# Run with current user
docker-compose run --user $(id -u):$(id -g) jupyter
```

### Can't access Jupyter
```bash
# Check if container is running
docker ps

# Check logs
docker-compose logs jupyter
```

### Out of disk space
```bash
# Clean up Docker
docker system prune -a
```

## Production Deployment

For production, modify Dockerfile:

```dockerfile
# Add authentication
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token='YOUR_SECRET_TOKEN'"]
```

## Multi-Stage Build (Optional)

For smaller images, use multi-stage build:

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
```

## Tips

1. **Data Persistence**: Always use volumes for data
2. **Development**: Mount source code with volumes
3. **Production**: Copy source code into image
4. **Security**: Don't run as root in production
5. **Resources**: Limit CPU/memory in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

## Next Steps

1. Add your data to `data/raw/`
2. Start Jupyter: `docker-compose up`
3. Open http://localhost:8888
4. Run notebooks sequentially
5. View results in `results/` folder

---

**Your project is now Dockerized!** 🐳
