'use client'

import { Button } from '@/components/ui/button'
import {
  setCaption,
  setEditorCanvasData,
  setEditorCanvasStatus,
  setSelectedImage,
  setMaskData,
  setMaskIndex,
  setSelectedMaskData,
  setMaskBox,
  setSelectionBox,
  setScaledSelectionBox,
  setObjectCaption,
  // setMasks,
  setSelectedMasks,
  nextMask,
  includeMask,
  excludeMask,
} from '@/lib/features/image-editor/imageEditorSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'
import { useAppDispatch, useAppSelector } from '@/lib/hooks'
import { controller } from '@/lib/utils'
import { queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  MouseEventHandler,
  PointerEventHandler,
  useEffect,
  useRef,
  useState,
  WheelEventHandler,
} from 'react'
import { ImageControlPanel } from './ImageControlPanel'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'

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
  const response = await fetch('http://127.0.0.1:8000/api/caption', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
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
  })
  return await response.json()
}

const fetchCroppedImageCaption = async (request: CroppedImageCaptionRequest) => {
  const response = await fetch('http://127.0.0.1:8000/api/cropped-image-caption', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  console.log(`response: ${JSON.stringify(response)}`)
  const json = await response.json()
  return json.caption
}

export const ImageEditor = () => {
  let selectedImage: MetadataEntry | null = useAppSelector(
    (state) => state.imageEditor.selectedImage
  )

  const tokenize = async (text: string) => {
    const response = await fetch('http://127.0.0.1:8000/api/tokenize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    })
    const { tokens } = await response.json()
    return tokens
  }

  const editorCanvasStatus = useAppSelector((state) => state.imageEditor.status)
  const maskIndex = useAppSelector((state) => state.imageEditor.maskIndex)
  const selectedMaskData = useAppSelector((state) => state.imageEditor.selectedMaskData)
  const maskBox = useAppSelector((state) => state.imageEditor.maskBox)
  const scaledSelectionBox = useAppSelector((state) => state.imageEditor.scaledSelectionBox)
  // const masks = useAppSelector((state) => state.imageEditor.masks)

  const [currentImage, setCurrentImage] = useState<HTMLImageElement | null>(null)
  const [captionScore, setCaptionScore] = useState(null)
  const [captionScoreLoading, setCaptionScoreLoading] = useState(false)
  const [maskVisible, setMaskVisible] = useState(false)
  const [selectionBoxVisible, setSelectionBoxVisible] = useState(false)
  const [masks, setMasks] = useState<
    { segmentation: HTMLImageElement; mask: HTMLImageElement; inverted_mask: HTMLImageElement }[]
  >([])
  // const [selectedMasks, setSelectedMasks] = useState<HTMLImageElement[]>([])

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef(1)
  const fetchingMask = useRef(false)
  const pointerCanvasCoords = useRef([0, 0])
  const pointerImageCoords = useRef([0, 0])
  const imageSize = useRef([0, 0])
  const oldCanvasDims = useRef([0, 0])
  const objectCaptionRef = useRef(null)

  const caption = useAppSelector((state) => state.imageEditor.caption)
  const objectCaption = useAppSelector((state) => state.imageEditor.objectCaption)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const maskData = useAppSelector((state) => state.imageEditor.maskData)
  const selectedMasks = useAppSelector((state) => state.imageEditor.selectedMasks)
  const selectionBox = useAppSelector((state) => state.imageEditor.selectionBox)

  const dispatch = useAppDispatch()

  const [tokenCount, setTokenCount] = useState(0)

  const queryClient = useQueryClient()

  const captionRef = useRef(null)

  // const selectionBox = useRef([0, 0, 0, 0])
  const pointerDownRef = useRef(false)
  const observerRef = useRef<any>({})
  const contextMenuOpen = useRef(false)

  const { data } = useQuery(
    queryOptions({
      queryKey: ['caption', selectedImage?.id],
      queryFn: async () => await get_caption(selectedImage as MetadataEntry),
      enabled: !!selectedImage,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  const { data: caption_score } = useQuery(
    queryOptions({
      queryKey: ['score', selectedImage?.id],
      queryFn: async () => {
        console.log(`filename: ${selectedImage?.filename}, caption: ${data.caption}`)
        const captionScore = await score({
          filename: selectedImage?.filename,
          caption: data.caption,
        })
        return captionScore
      },
      enabled: captionScoreLoading,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  const scoreButtonClickHandler: MouseEventHandler<HTMLButtonElement> = async () => {
    if (captionScoreLoading) return
    setCaptionScoreLoading(true)
  }

  const getCroppedImageCaption = async () => {
    console.log(`scaleRef.current: ${scaleRef.current}`)

    if (!selectedImage) return
    const [x, y, w, h] = scaledSelectionBox

    const result = await fetchCroppedImageCaption({
      image_data: selectedImage?.image_data as string,
      crop_box: [x, y, x + w, y + h],
    })
    console.log(`result:`, result)
    return result
  }

  const drawCurrentImage = () => {
    if (!canvasRef.current) return
    console.log(`drawCurrentImage`)
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    if (currentImage) {
      ctx.clearRect(0, 0, canv.width, canv.height)
      const scaleX = canv.width / currentImage.width
      const scaleY = canv.height / currentImage.height
      const scale = Math.min(scaleX, scaleY)
      ctx.drawImage(
        currentImage,
        0,
        0,
        currentImage.width,
        currentImage.height,
        canv.width / 2 - (currentImage.width / 2) * scale,
        canv.height / 2 - (currentImage.height / 2) * scale,
        currentImage.width * scale,
        currentImage.height * scale
      )
    }
  }

  const drawMasks = async () => {
    if (!canvasRef.current) return
    console.log(`drawMasks`)
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    if (masks.length && maskVisible) {
      for (const index in maskData) {
        const [x, y, w, h] = maskData[index].canvas_box

        let image = new Image()
        if (maskData[index].active) {
          image = masks[index].segmentation
        }
        if (maskData[index].include) {
          image = masks[index].mask
        }
        if (maskData[index].exclude) {
          image = masks[index].inverted_mask
        }
        if (!image) continue
        console.log(`image.width: ${image.width} image.height: ${image.height}`)

        if (!image.complete) await image.decode()
        ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
      }
    }
  }

  const drawSelectionBox = () => {
    if (!canvasRef.current) return
    console.log(`drawSelectionBox`)
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    const [x, y, w, h] = selectionBox

    if (selectionBoxVisible) {
      ctx.strokeStyle = 'white'
      ctx.setLineDash([15, 15])
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
    }
  }

  const pointerDown: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current || contextMenuOpen.current || !currentImage || pointerDownRef.current)
      return
    console.log(`pointerDown`)
    if (fetchingMask.current) {
      drawSelectionBox()
      return
    }
    const canv = canvasRef.current as HTMLCanvasElement
    pointerDownRef.current = true
    const { x: bx, y: by } = canv.getBoundingClientRect()
    const x = e.clientX - bx
    const y = e.clientY - by
    pointerCanvasCoords.current = [x, y, 0, 0]
    dispatch(setSelectionBox([x, y, 0, 0]))
    setSelectionBoxVisible(true)
  }

  const pointerMove: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current) return
    console.log(`pointerMove`)
    if (fetchingMask.current) {
      drawCurrentImage()
      drawMasks()
      drawSelectionBox()
      return
    }
    const canv = canvasRef.current as HTMLCanvasElement

    if (pointerDownRef.current && !contextMenuOpen.current) {
      const [x1, y1] = selectionBox
      const { x: bx, y: by } = canv.getBoundingClientRect()
      const x2 = e.clientX - bx
      const y2 = e.clientY - by
      const x = Math.min(x1, x2)
      const y = Math.min(y1, y2)
      const w = Math.abs(x2 - x1)
      const h = Math.abs(y2 - y1)

      dispatch(setSelectionBox([x, y, w, h]))
      if (currentImage) {
        const scaleX = canv.width / currentImage?.width
        const scaleY = canv.height / currentImage.height
        const scale = Math.min(scaleX, scaleY)
        dispatch(setScaledSelectionBox([x / scale, y / scale, w / scale, h / scale]))
      }
      const ctx = canv.getContext('2d')
      if (!ctx) return

      drawCurrentImage()
      drawMasks()
      drawSelectionBox()
    }
  }

  const pointerUp: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current || contextMenuOpen.current || !pointerDownRef.current) return
    console.log(`pointerUp`)
    if (fetchingMask.current) {
      drawSelectionBox()
      return
    }

    const canv = canvasRef.current as HTMLCanvasElement
    pointerDownRef.current = false

    const [x1, y1] = selectionBox
    const { x: bx, y: by } = canv.getBoundingClientRect()
    const x2 = e.clientX - bx
    const y2 = e.clientY - by
    const x = Math.min(x1, x2)
    const y = Math.min(y1, y2)
    const w = Math.abs(x2 - x1)
    const h = Math.abs(y2 - y1)

    if (Math.abs(x2 - x) < 10 && Math.abs(y2 - y) < 10 && maskData.length) {
      console.log(`maskData: `, maskData)
      dispatch(nextMask())
      console.log(`maskIndex: ${maskIndex.value}`)
      dispatch(setMaskIndex((maskIndex.value + 1) % maskData.length))
      dispatch(setSelectionBox([0, 0, canv.width, canv.height]))
    } else {
      dispatch(setSelectionBox([x, y, w, h]))
      dispatch(setMaskBox([x, y, w, h]))
    }

    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }

  const removeSelect = () => {
    dispatch(
      setSelectedMaskData(
        selectedMaskData.filter(
          (data) =>
            data.point_coords[0] !== maskData[maskIndex.value].point_coords[0] &&
            data.point_coords[1] !== maskData[maskIndex.value].point_coords[1] &&
            data.area !== maskData[maskIndex.value].area
        )
      )
    )
  }

  const maskItemSelectHandler = (e) => {
    console.log('maskItemSelectHandler')
    e.stopPropagation()
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
    const [x, y, w, h] = maskBox
    console.log(`maskBox: ${maskBox}`)
    // if (w < 10 && h < 10) return
    // drawCurrentImage()
    if (!canvasRef.current || !currentImage) return
    const canv = canvasRef.current as HTMLCanvasElement

    const scaleX = canv.width / currentImage.width
    const scaleY = canv.width / currentImage.height
    const scale = Math.min(scaleX, scaleY)

    const [sx, sy] = [
      Math.floor((x - (canv.width / 2 - (currentImage.width * scale) / 2)) / scale),
      Math.floor((y - (canv.height / 2 - (currentImage.height * scale) / 2)) / scale),
    ]

    const [sw, sh] = [Math.floor(w / scale), Math.floor(h / scale)]
    const bbox = [sx, sy, sx + sw, sy + sh]

    console.log(`bbox: ${bbox}`)

    const ctx = canv.getContext('2d')
    if (!ctx) return
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()

    const ws = new WebSocket('ws://127.0.0.1:8000/ws/mask')

    if (!selectedImage) return

    const message = {
      image: selectedImage.image_data,
      bbox: bbox,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
      fetchingMask.current = false
    }
    ws.onclose = (event) => {
      fetchingMask.current = false
      console.log('WebSocket closed', event)
      setSelectionBoxVisible(false)
      dispatch(toggleDisabled(false))
    }

    ws.onopen = () => {
      ws.send(JSON.stringify(message))
      setSelectionBoxVisible(true)
      fetchingMask.current = true
      drawSelectionBox()
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (!Object.hasOwn(data, 'status')) {
          console.log(`data: `, data)
          let old_data = [...maskData]
          old_data = old_data.map((d) => ({ ...d, active: false }))
          let new_data = [...data]
          new_data = new_data.map((d) => ({ ...d, canvas_box: maskBox, bbox: bbox, active: false }))
          const newMaskData = [...old_data, ...new_data].toSorted((d1, d2) => d2.area - d1.area)
          dispatch(setMaskData(newMaskData))
          dispatch(setMaskIndex(0))
          if (canvasRef.current) {
            const canv = canvasRef.current as HTMLCanvasElement
            const ctx = canv.getContext('2d')
            if (!ctx) return
            const image = new Image()
            image.src = `data:image/png;base64,${newMaskData[0].segmentation}`
            image.onload = () => {
              const [x, y, w, h] = new_data[0].canvas_box
              ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)

              setMaskVisible(true)
            }
          }
          ws.close()
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
  }

  // useEffect(() => {
  //   if (!canvasRef.current) return
  //   const canv = canvasRef.current as HTMLCanvasElement

  //   fetch(`http://127.0.0.1:8000/api/images`, {
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //   })
  //     .then((res) => res.json())
  //     .then((data) => {
  //       const image = data.find((item: any) => item.original_filename === 'profile-picture.jpg')
  //       if (image) {
  //         dispatch(setSelectedImage(image))
  //       }
  //     })
  //     .catch((e) => console.error(e))
  // }, [])

  useEffect(() => {
    if (!maskData.length) return
    console.log(`maskData: `, maskData)

    const maskImages: MaskImage[] = []

    ;(async () => {
      for (let data of maskData) {
        data = data as MaskData

        let b64 = data.segmentation
        let segmentation = new Image()
        segmentation.src = `data:image/png;base64,${b64}`
        // await segmentation.decode()

        b64 = data.mask
        const mask = new Image()
        mask.src = `data:image/png;base64,${b64}`
        // await mask.decode()

        b64 = data.inverted_mask
        const inverted_mask = new Image()
        inverted_mask.src = `data:image/png;base64,${b64}`
        // await inverted_mask.decode()

        maskImages.push({ segmentation, mask, inverted_mask })
      }

      setMasks([...maskImages])
      setMaskVisible(true)
    })()
  }, [maskData])

  useEffect(() => {
    console.log(`useEffect selectedMaskData`)
    if (!selectedMaskData.length || !canvasRef.current || !currentImage) return
    console.log(`selectedMaskData: `, selectedMaskData)
    const masks: HTMLImageElement[] = []
    for (const data of selectedMaskData) {
      const image = new Image()
      image.src = `data:image/png;base64,${data.mask}`
      image.onload = () => {
        masks.push(image)
      }
    }
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    for (const index in selectedMaskData) {
      const [x, y, w, h] = selectedMaskData[index].canvas_box
      const mask = masks[index]
      if (mask.complete) {
        ctx.drawImage(mask, 0, 0, mask.width, mask.height, x, y, w, h)
      } else {
        mask.onload = () => {
          ctx.drawImage(mask, 0, 0, mask.width, mask.height, x, y, w, h)
        }
      }
    }
    setSelectedMasks(masks)
  }, [selectedMaskData])

  // useEffect(() => {
  //   if (!selectedMasks.length || !canvasRef.current) return
  //   console.log(`selectedMasks: `, selectedMasks)
  //   const canv = canvasRef.current
  //   const ctx = canv.getContext('2d')
  //   if (!ctx) return

  //   for (const index in selectedMasks) {
  //     const [x, y, w, h] = selectedMaskData[index].canvas_box
  //     const mask = selectedMasks[index]
  //     let image = new Image()
  //     if (maskData[index].active) {
  //       image = masks[index].segmentation
  //     }
  //     if (maskData[index].include) {
  //       image = masks[index].mask
  //     }
  //     if (maskData[index].exclude) {
  //       image = masks[index].inverted_mask
  //     }

  //     ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
  //   }
  // }, [selectedMasks])

  useEffect(() => {
    if (!selectedImage) return
    console.log(`useEffect selectedImage`)
    const b64 = selectedImage.image_data
    const url = `data:image/png;base64,${b64}`
    const image = new Image()
    image.src = url
    image.onload = () => {
      setCurrentImage(image)
    }
  }, [selectedImage])

  useEffect(() => {
    if (!canvasRef.current || !currentImage) return
    console.log(`useEffect currentImage`)
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    const image = currentImage

    const size = Math.min(canv.width, canv.height)
    const scale_x = size / image.width
    const scale_y = size / image.height
    let scale = Math.min(scale_x, scale_y)
    ctx.clearRect(0, 0, canv.width, canv.height)
    ctx.drawImage(
      image,
      0,
      0,
      image.width,
      image.height,
      canv.width / 2 - (image.width / 2) * scale,
      canv.height / 2 - (image.height / 2) * scale,
      image.width * scale,
      image.height * scale
    )

    drawMasks()

    const observer = new ResizeObserver(async (entries, target) => {
      const width = entries[0].borderBoxSize[0].inlineSize
      const height = entries[0].borderBoxSize[0].blockSize

      const size = Math.min(width, height)
      const scale_x = size / image.width
      const scale_y = size / image.height
      let scale = Math.min(scale_x, scale_y)
      canv.width = width
      canv.height = height

      const ctx = canv.getContext('2d')
      drawCurrentImage()
      drawMasks()

      let [x, y, w, h] = selectionBox

      let scaled_x = Math.min(width, 1024) / currentImage.width
      let scaled_y = Math.min(height, 1024) / currentImage.height
      scale = Math.min(scaled_x, scaled_y)

      dispatch(setSelectionBox([x, y, w, h]))
      dispatch(setMaskBox([x, y, w, h]))

      dispatch(
        setEditorCanvasStatus({
          ...editorCanvasStatus,
          canvas: `width: ${canv.width}, height: ${canv.height}`,
          image: `width: ${image.width}, height: ${image.height}`,
          scale: scale,
        })
      )
    })

    observer.observe(canv)

    return () => {
      observer.disconnect()
    }
  }, [currentImage])

  useEffect(() => {
    if (!masks.length) return
    console.log(`useEffect masks`)
    console.log(`masks: `, masks)
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }, [masks, currentImage, selectionBox])

  useEffect(() => {
    if (!maskData.length) return
    console.log(`useEffect maskIndex`)
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }, [maskIndex])

  // useEffect(() => {
  //   if (!maskData || !maskData.length) return
  //   console.log(`maskData: `, maskData)
  //   console.log(`maskIndex: `, maskIndex)
  //   console.log(`maskIndex.value: ${maskIndex.value}`)
  //   const dataUrl = `data:image/png;base64,${maskData[maskIndex.value].segmentation}`
  //   const image = new Image()
  //   image.src = dataUrl
  //   image.onload = () => setMask(image)
  // }, [maskData])

  useEffect(() => {
    if (data) {
      dispatch(setCaption(data.caption))
    }
  }, [data])

  useEffect(() => {
    if (captionRef.current) {
      ;(captionRef.current as HTMLTextAreaElement).value = caption || ''
    }
  }, [caption])

  useEffect(() => {
    if (!caption_score) return
    setCaptionScore(caption_score.score || 0)
    setCaptionScoreLoading(false)
  }, [caption_score])

  return (
    <ResizablePanelGroup
      direction="vertical"
      className="flex flex-col items-center justify-center w-full h-full border-2 border-solid border-neutral-950"
    >
      <ResizablePanel className="flex flex-col w-full h-full min-h-[48px] items-center justify-center bg-neutral-900  p-2">
        <h1 className="text-lg font-bold text-white w-full">
          {selectedImage?.original_filename as string}
        </h1>
        <div className="flex flex-col relative justify-start w-full">
          <ContextMenu
            onOpenChange={(open) => {
              contextMenuOpen.current = open
              pointerDownRef.current = false
            }}
          >
            <ContextMenuTrigger>
              <canvas
                ref={canvasRef}
                className="relative w-full h-full cursor-crosshair aspect-square"
                width={1024}
                height={1024}
                onPointerDown={pointerDown}
                onPointerMove={pointerMove}
                onPointerUp={pointerUp}
                // onWheel={canvasWheelHandler}
              ></canvas>
            </ContextMenuTrigger>
            <ContextMenuContent className="bg-neutral-800">
              <ContextMenuItem
                onSelect={(e) => {
                  drawSelectionBox()
                  maskItemSelectHandler(e)
                }}
              >
                Mask
              </ContextMenuItem>
              <ContextMenuItem
                disabled={!maskData.length}
                onSelect={(e) => {
                  e.stopPropagation()
                  setMaskVisible(!maskVisible)
                }}
              >
                {maskVisible ? 'Hide' : 'Show'}
              </ContextMenuItem>
              <ContextMenuItem
                disabled={!maskData.length}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(includeMask(maskIndex.value))
                }}
              >
                Include
              </ContextMenuItem>
              <ContextMenuItem
                disabled={!maskData.length}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(excludeMask(maskIndex.value))
                  let newSelectedMasks = [...selectedMasks]
                  const { id, inverted_mask } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData: inverted_mask })
                  } else {
                    newSelectedMasks = newSelectedMasks.map((mask, index) => {
                      if (index === maskIndex.value) {
                        return { id, imageData: inverted_mask }
                      }
                      return mask
                    })
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Exclude
              </ContextMenuItem>
              <ContextMenuItem
                disabled={!maskData.length}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex.value) {
                          new_data.active = true
                          new_data.include = false
                          new_data.exclude = false
                        }
                        return new_data
                      })
                    )
                  )
                  const newSelectedMasks = [...selectedMasks]
                  const { id, segmentation } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData: segmentation })
                  } else {
                    newSelectedMasks.splice(maskIndex.value, 1, { id, imageData: segmentation })
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Deselect
              </ContextMenuItem>
              <ContextMenuItem
                onSelect={() => {
                  dispatch(setSelectedMasks([]))
                  setMasks([])
                  dispatch(setMaskData([]))
                }}
              >
                Clear
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
          <div className="h-[2rem] w-full bg-neutral-800 text-white text-sm p-2 m-0 flex gap-x-4 overflow-clip">
            {Object.entries(editorCanvasStatus).map((entry) => (
              <span key={entry[0]}>{`${entry[0]}: ${entry[1]}\t`}</span>
            ))}
          </div>
        </div>
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel className="flex flex-col items-start justify-around w-full h-full bg-neutral-900">
        <div className="flex items-center justify-center space-x-4 w-full">
          <Button
            size="lg"
            aria-label="merge select"
            onClick={() => {
              const selectedMask = { ...maskData[maskIndex.value] }
              const [x, y, w, h] = scaledSelectionBox
              selectedMask.bbox = [x, y, x + w, y + h]
              selectedMask.canvas_box = maskBox
              dispatch(setSelectedMaskData([...selectedMaskData, selectedMask]))
              dispatch(
                setMaskData([
                  ...maskData.map((data, index) => {
                    if (index === maskIndex.value) {
                      data.include == true
                    }
                    return data
                  }),
                ])
              )
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="48px"
              viewBox="0 -960 960 960"
              width="48px"
              fill="#e8eaed"
            >
              <path d="M400-80q-33 0-56.5-23.5T320-160v-160H160q-33 0-56.5-23.5T80-400v-400q0-33 23.5-56.5T160-880h400q33 0 56.5 23.5T640-800v160h160q33 0 56.5 23.5T880-560v400q0 33-23.5 56.5T800-80H400Zm0-80h400v-400H560v-240H160v400h240v240Zm80-320Z" />
            </svg>
          </Button>
          <Button size="lg" aria-label="remove select" onClick={removeSelect}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="48px"
              viewBox="0 -960 960 960"
              width="48px"
              fill="#e8eaed"
            >
              <path d="m500-120-56-56 142-142-142-142 56-56 142 142 142-142 56 56-142 142 142 142-56 56-142-142-142 142Zm-220 0v-80h80v80h-80Zm-80-640h-80q0-33 23.5-56.5T200-840v80Zm80 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80q33 0 56.5 23.5T840-760h-80ZM200-200v80q-33 0-56.5-23.5T120-200h80Zm-80-80v-80h80v80h-80Zm0-160v-80h80v80h-80Zm0-160v-80h80v80h-80Zm640 0v-80h80v80h-80Z" />
            </svg>
          </Button>
        </div>
        <textarea
          ref={captionRef}
          className="text-lg text-white w-full h-full m-0 px-2 resize-none border-2 border-solid border-neutral-800"
          defaultValue={caption || 'Loading...'}
          onChange={async (e) => {
            dispatch(setCaption(e.target.value))
            const count = await tokenize(e.target.value)
            setTokenCount(count.length)
          }}
        />
        <div className="flex flex-row items-center h-32 w-full px-2 m-0">
          <p
            className="bg-neutral-900 w-full"
            style={{ color: tokenCount <= 77 ? 'white' : 'mediumvioletred' }}
          >
            tokens: {tokenCount} / 77
          </p>
        </div>
        {/*<textarea
          ref={objectCaptionRef}
          className="text-lg text-white w-4/5 h-full m-0 px-2 resize-none border-2 border-solid border-neutral-800"
          defaultValue={objectCaption}
          onChange={(e) => dispatch(setObjectCaption(e.target.value))}
        />*/}
        {/* <div className="flex flex-col items-center justify-between h-full w-1/5 p-2 border-2 border-solid border-neutral-800">
          <h1 className="text-lg font-bold text-white">
            Match Score:{' '}
            {captionScoreLoading
              ? 'Loading...'
              : captionScore
                ? `${(captionScore as number).toFixed(2)}%`
                : (0).toFixed(2)}
          </h1>
          <Button onClick={scoreButtonClickHandler}>Score</Button>
        </div> */}
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
