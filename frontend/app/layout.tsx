import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'

export const metadata: Metadata = {
  title: '',
  description: '',
  keywords: [],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="min-[0px]:w-screen min-[0px]:h-screen">
      <body className={'min-[0px]:w-full min-[0px]:h-full md:w-screen md:h-screen antialiased'}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
