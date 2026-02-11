import type { Metadata } from 'next'
import './globals.css'
import Sidebar from '@/components/Sidebar'
import { PreviewProvider } from '@/context/PreviewContext'
import RightPanelWrapper from '@/components/RightPanelWrapper'

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
      <body className="antialiased h-screen w-full overflow-hidden bg-white">
        <PreviewProvider>
          <div className="flex h-full w-full overflow-hidden">
            {/* Sidebar - Fixed */}
            <div className="shrink-0 flex flex-col h-full border-r border-gray-100 shadow-sm z-20">
              <Sidebar />
            </div>

            {/* Main Content Area - Flex container for Chat + Preview */}
            <div className="flex-1 flex overflow-hidden bg-slate-50/20 relative">
              {/* Chat View - Expands/Shrinks */}
              <div className="flex-1 min-w-0 flex flex-col relative bg-white overflow-hidden shadow-inner">
                {children}
              </div>

              {/* Preview Panel - Pushes Chat to the left */}
              <RightPanelWrapper />
            </div>
          </div>
        </PreviewProvider>
      </body>
    </html>
  )
}
