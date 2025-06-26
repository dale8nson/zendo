'use client'

import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { useEffect, useRef, useState } from 'react'
import type { MetadataEntry } from './ImageGallery'
import { useQuery, queryOptions, useQueryClient } from '@tanstack/react-query'
import { setSelectedImage, setCaption } from '@/lib/features/image-editor/imageEditorSlice'
import { controller } from '@/lib/utils'

interface ScoreRequest {
  filename: string | undefined
  caption: string | null
}

const predict = async (data: MetadataEntry | null) => {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  return await response.json()
}

const get_caption = async (data: MetadataEntry) => {
  const response = await fetch('/api/caption', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
    signal: controller.signal,
  })
  return await response.json()
}

const score = async (data: ScoreRequest | null) => {
  const response = await fetch('/api/score', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
    signal: controller.signal,
  })
  return await response.json()
}

export const ImageEditor = () => {
  let selectedImage: MetadataEntry | null = useAppSelector((state) => state.imageEditor.image)
  const canvas = useRef<HTMLCanvasElement>(null)

  const caption = useAppSelector((state) => state.imageEditor.caption)
  const dispatch = useAppDispatch()
  const queryClient = useQueryClient()

  const textRef = useRef(null)
  // const [caption, setCaption] = useState('')

  const { data, isLoading } = useQuery(
    queryOptions({
      queryKey: ['caption', selectedImage?.id],
      queryFn: async () => await get_caption(selectedImage as MetadataEntry),
      enabled: !!selectedImage,
      refetchOnWindowFocus: false,
    })
  )

  const { data: caption_score, isLoading: captionScoreLoading } = useQuery(
    queryOptions({
      queryKey: ['score', selectedImage?.id],
      queryFn: async () => {
        console.log(`filename: ${selectedImage?.filename}, caption: ${data.caption}`)
        return await score({ filename: selectedImage?.filename, caption: data.caption })
      },
      enabled: !!data,
      refetchOnWindowFocus: false,
    })
  )

  const [captionScore, setCaptionScore] = useState(null)

  const bounce = useRef(false)

  const captionChangeHandler = async (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (bounce.current) return
    bounce.current = true
    dispatch(setCaption(e.target.value))
    const result = await score({
      filename: selectedImage?.filename,
      caption: e.target.value,
    })
    if (textRef.current) (textRef.current as HTMLTextAreaElement).value = e.target.value
    setCaptionScore(result.score)
    setTimeout(() => {
      bounce.current = false
    }, 500)
  }

  useEffect(() => {
    if (!selectedImage || !canvas.current) return
    selectedImage = selectedImage as MetadataEntry
    queryClient.invalidateQueries({ queryKey: ['caption', selectedImage?.id] })
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
      console.log(`canvas.width: ${canv.width}, canvas.height: ${canv.height}`)
      const width_scale = canv.width / image.width
      const height_scale = canv.height / image.height
      const scale = Math.min(width_scale, height_scale)
      console.log(`scale: ${scale}`)
      // ctx?.scale(scale, scale)
      ctx?.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        canv.width / 2 - (image.width * scale) / 2,
        canv.height / 2 - (image.height * scale) / 2,
        image.width * scale,
        image.height * scale
      )
    }

    image.onload = () => {
      drawImage()
    }

    const observer = new ResizeObserver((entries, target) => {
      canv.width = entries[0].borderBoxSize[0].inlineSize
      canv.height = entries[0].borderBoxSize[0].blockSize
      drawImage()
    })
    observer.observe(canvas.current)

    image.onerror = () => {
      console.error('Failed to load image')
    }
  }, [selectedImage])

  useEffect(() => {
    if (data) {
      dispatch(setCaption(data.caption))
    }
  }, [data])

  useEffect(() => {
    if (textRef.current) {
      ;(textRef.current as HTMLTextAreaElement).value = caption || ''
    }
  }, [caption])

  useEffect(() => {
    if (!caption_score) return
    setCaptionScore(caption_score.score || 0)
  }, [caption_score])

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
          <div className="flex items-start justify-around w-full bg-neutral-900">
            <textarea
              ref={textRef}
              className="text-2xl font-bold text-white w-4/5 h-full m-0 px-2 resize-none border-2 border-solid border-neutral-800"
              defaultValue={caption || 'Loading...'}
              onChange={captionChangeHandler}
            />
            <div className="flex items-center justify-center h-full w-1/5 p-2 border-2 border-solid border-neutral-800">
              <h1 className="text-xl font-bold text-white">
                Match Score:{' '}
                {captionScore
                  ? `${(captionScore as number).toFixed(2)}%`
                  : captionScoreLoading
                    ? 'Loading...'
                    : (0).toFixed(2)}
              </h1>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}
