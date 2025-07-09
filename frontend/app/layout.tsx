import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'
import { Toaster } from 'sonner'
import { controller } from '@/lib/utils'

export const metadata: Metadata = {
  title: 'ZendoAI',
  description: 'AI-powered image editing and transformation tool',
  keywords: ['AI', 'image', 'editing', 'transformation', 'tool'],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  controller.abort()
  return (
    <html lang="en" className="w-screen h-screen">
      <body className={'w-screen h-screen md:w-screen md:h-screen antialiased'}>
        <Providers>
          {children}
          <Toaster position="top-center" />
        </Providers>
      </body>
    </html>
  )
}
