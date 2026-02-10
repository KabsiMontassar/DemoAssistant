export const getApiUrl = () => {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    // If we're in Azure (not localhost), point to the same host but port 8000
    if (hostname !== 'localhost' && hostname !== '127.0.0.1' && !hostname.startsWith('192.168.') && !hostname.startsWith('10.')) {
      return `http://${hostname}:8000`;
    }
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
}

export const API_BASE_URL = getApiUrl();
