# Docker Setup Guide

This project is containerized with Docker and can be run using Docker Compose.

## Prerequisites

- Docker (v20.10+)
- Docker Compose (v2.0+)

## Quick Start

### 1. Setup Environment Variables

Copy the example environment file and update it with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your `OPENROUTER_API_KEY`:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Build and Run with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services in the background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Individual Service Commands

### Build Only

```bash
# Build backend
docker-compose build backend

# Build frontend
docker-compose build frontend
```

### Run Specific Service

```bash
# Run only backend
docker-compose up -d backend

# Run only frontend
docker-compose up -d frontend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Execute Commands in Container

```bash
# Backend shell
docker-compose exec backend /bin/bash

# Frontend shell
docker-compose exec frontend /bin/sh
```

## Docker Compose Configuration

The `docker-compose.yml` includes:

- **backend**: Python FastAPI service on port 8000
  - Volumes: Data persistence for materials and embeddings
  - Health check: Monitors API availability
  - Environment variables: API keys and paths

- **frontend**: Next.js service on port 3000
  - Production build with multi-stage build
  - API URL configuration
  - Auto-restart policy

- **Network**: Shared bridge network for service communication

## Building Standalone Docker Images

### Build Backend Image

```bash
cd backend
docker build -t material-pricing-backend:latest .
```

### Build Frontend Image

```bash
cd frontend
docker build -t material-pricing-frontend:latest .
```

### Run Standalone

```bash
# Backend
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  -v $(pwd)/backend/data:/app/data \
  material-pricing-backend:latest

# Frontend
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  material-pricing-frontend:latest
```

## Troubleshooting

### Services not communicating

- Check network: `docker-compose ps`
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check logs: `docker-compose logs`

### Data persistence

- Backend data is stored in `./backend/data` volume
- Delete the volume to reset: `docker-compose down -v`

### Rebuilding after code changes

```bash
# Remove and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Clearing Docker resources

```bash
# Stop all containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Clean volumes
docker volume prune
```

## Environment Variables

### Backend (docker-compose.yml)

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `DATA_PATH`: Path to materials data (default: /app/data)
- `CHROMA_PATH`: Path to Chroma DB (default: /app/data/chroma_db)
- `BACKEND_HOST`: Server host binding (default: 0.0.0.0)
- `BACKEND_PORT`: Server port (default: 8000)

### Frontend (docker-compose.yml)

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:8000)
- `NODE_ENV`: Environment mode (set to 'production' in Docker)

## Deployment Notes

For production deployment:

1. Update API URLs to production endpoints
2. Use environment-specific `.env` files
3. Consider using a reverse proxy (nginx)
4. Use proper secret management instead of .env files
5. Set up persistent volumes for data storage
6. Configure proper logging and monitoring

## Development with Docker

To make changes while containers are running:

```bash
# Rebuild specific service
docker-compose build backend

# Restart service
docker-compose up -d backend
```

Note: Frontend uses Next.js build output, so changes require a rebuild. Backend uses Python with file watching, so some changes may hot-reload.
