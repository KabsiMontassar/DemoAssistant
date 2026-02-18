export const getApiUrl = () => {
  // All API calls are proxied through the Next.js server to the backend.
  // This keeps the backend off the public internet.
  return '';
}

export const API_BASE_URL = getApiUrl();
