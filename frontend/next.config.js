/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/health',
        destination: 'http://atlas-backend:8000/health',
      },
      {
        source: '/api/:path*',
        destination: 'http://atlas-backend:8000/api/:path*',
      },
    ];
  },
}

module.exports = nextConfig
