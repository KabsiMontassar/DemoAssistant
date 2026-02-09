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
      <body className="antialiased h-screen w-full overflow-hidden">
        <div className="flex  h-full">
          <Sidebar />
          <main className="flex-1 relative h-full overflow-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
