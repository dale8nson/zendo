'use client'

import { useAppSelector } from '@/lib/hooks'
import { useEffect, useRef, useState } from 'react'
import type { MetadataEntry } from './ImageGallery'
import { useQuery, queryOptions } from '@tanstack/react-query'

const predict = async (data: MetadataEntry | null) => {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

const caption = async (data: MetadataEntry | null) => {
  const response = await fetch('/api/caption', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

export const ImageEditor = () => {
  let selectedImage: MetadataEntry | null = useAppSelector((state) => state.imageEditor.image)
  const canvas = useRef<HTMLCanvasElement>(null)

  const { data, isLoading } = useQuery(
    queryOptions({
      queryKey: ['prediction', selectedImage?.id],
      queryFn: () => caption(selectedImage ? selectedImage : null),
    })
  )

  useEffect(() => {
    if (!selectedImage || !canvas.current) return
    selectedImage = selectedImage as MetadataEntry
    const canv = canvas.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    const { width, height } = canvas.current.getBoundingClientRect()
    console.log(`width: ${canv.clientWidth}, height: ${canv.clientHeight}`)
    canv.width = width
    canv.height = height
    console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
    const image = new Image(selectedImage.width, selectedImage.height)
    console.log(`image.width: ${image.width}, image.height: ${image.height}`)
    image.src = `data:image/${selectedImage.filename.split('.').pop()};base64,${selectedImage.image_data}`
    console.log(
      `selectedImage.width: ${selectedImage.width}, selectedImage.height: ${selectedImage.height}`
    )

    const drawImage = () => {
      console.log(`drawImage`)
      ctx?.clearRect(0, 0, width, height)
      ctx?.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        canv.width / 2 - ((canv.width / image.width) * image.width) / 2,
        canv.height / 2 - ((canv.width / image.width) * image.height) / 2,
        canv.width,
        (canv.width / image.width) * image.height
      )
    }

    image.onload = () => {
      drawImage()
    }

    const observer = new ResizeObserver((entries, target) => {
      // console.log(
      //   `canvas.width: ${target || canv.width}, canvas.height: ${canvas.current.height || canv.height}`
      // )
      canv.width = entries[0].borderBoxSize[0].inlineSize
      canv.height = entries[0].borderBoxSize[0].blockSize
      drawImage()
    })
    observer.observe(canvas.current)

    image.onerror = () => {
      console.error('Failed to load image')
    }
  }, [selectedImage])

  if (data) {
    console.log(data)
  }

  return (
    <>
      {selectedImage ? (
        <div className=" absolute top-0 left-0 flex flex-col items-center justify-center w-full h-full  border-2 border-solid border-neutral-950 bg-black/80">
          <div className="flex items-center justify-center w-full bg-neutral-900  p-2">
            <h1 className="text-2xl font-bold text-white">
              {selectedImage?.original_filename as string}
            </h1>
          </div>
          <canvas ref={canvas} className="top-0 left-0 w-full h-full" />
          <div className="flex items-center justify-center w-full bg-neutral-900  p-2">
            <h1 className="text-2xl font-bold text-white">{data ? data.caption : 'Loading...'}</h1>
          </div>
        </div>
      ) : null}
    </>
  )
}
