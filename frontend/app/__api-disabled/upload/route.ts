import { NextRequest, NextResponse } from 'next/server'

export const POST = async (req: NextRequest) => {
  const formData = await req.formData()
  const file = formData.get('file') as File
  const label = formData.get('label') as string

  const uploadFormData = new FormData()
  uploadFormData.append('file', file)
  uploadFormData.append('label', label)

  const res = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: uploadFormData,
  })

  const data = await res.json()
  console.log(res)

  return NextResponse.json(data)
}
