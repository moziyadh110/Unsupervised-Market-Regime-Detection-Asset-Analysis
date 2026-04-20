# Docker Setup (NEW!)

## 🐳 Quick Start with Docker

The easiest way to run this project is using Docker:

```bash
# 1. Build and start
docker-compose up

# 2. Open http://localhost:8888 in your browser
# Jupyter Lab is ready to use!

# 3. Stop when done
docker-compose down
```

### Why Docker?

✅ No dependency conflicts
✅ Works on any OS (Windows, Mac, Linux)
✅ Isolated environment
✅ One command to start

### Docker Commands

```bash
# Start Jupyter Lab
make docker-up

# Run tests
make docker-test

# Open shell in container
make docker-shell

# Clean up
make docker-down
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md)

---

