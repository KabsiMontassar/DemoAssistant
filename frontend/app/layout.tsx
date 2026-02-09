import type { Metadata } from 'next'
import './globals.css'
import Sidebar from '@/components/Sidebar'

export const metadata: Metadata = {
  title: 'Atlas',
  description: 'Proffessional AI Material Assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <div className="main-app-container">
          <Sidebar />
          <main className="flex-1 relative overflow-hidden bg-white/50 backdrop-blur-sm z-10 transition-all duration-300">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
