import { NextRequest, NextResponse } from 'next/server'

export const GET = async (req: NextRequest) => {
  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/images`, {
      headers: { 'Access-Control-Allow-Origin': '*' },
    })
    if (!res.ok) {
      throw new Error(`Failed to fetch images: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()

    return NextResponse.json(data)
  } catch (error) {
    console.error(error)
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
