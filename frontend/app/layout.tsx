import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Material Pricing AI Assistant',
  description: 'RAG-based AI assistant for material pricing information',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
