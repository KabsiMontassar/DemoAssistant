# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with memory-saving settings
RUN npm config set progress false && \
    NODE_OPTIONS="--max-old-space-size=512" npm ci --no-audit --no-fund

# Copy source code
COPY . .

# Build Next.js app with memory limits
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN NODE_OPTIONS="--max-old-space-size=768" npm run build

# Production stage
FROM node:18-alpine AS runner

WORKDIR /app

# Set environment
ENV NODE_ENV=production

# Copy necessary files
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/next.config.js ./next.config.js

# Expose port
EXPOSE 3000

# Run application
CMD ["npm", "start"]
