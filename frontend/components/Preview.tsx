'use client'
import {
  use,
  useEffect,
  useRef,
  useState,
  useCallback,
  PointerEventHandler,
  WheelEventHandler,
} from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { useQuery } from '@tanstack/react-query'
import { controller } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'

import { ToolPalette } from './ToolPalette'

import {
  setCurrentHistoryIndex,
  setPreviewCanvasData,
  setShouldDrawCanvas,
  appendHistory,
  setPreviewStatus,
  setScaledSelectionBox,
  setMaskData,
  setMaskIndex,
  setMaskBox,
  setSelectedMaskData,
  setSelectedMasks,
  nextMask,
  includeMask,
  excludeMask,
} from '@/lib/features/preview/previewSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'
import { PromptPanel } from '@/components/PromptPanel'
import { Progress } from './ui/progress'
import { setEditorCanvasStatus } from '@/lib/features/image-editor/imageEditorSlice'

export function Preview() {
  console.log('Preview')

  const canvasRef = useRef(null)
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const iterations = useAppSelector((state) => state.controlPanel.generationIterations)
  const scaleRef = useRef(1)
  const fetchingMask = useRef(false)

  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const maskData = useAppSelector((state) => state.preview.maskData)
  const maskBox = useAppSelector((state) => state.preview.maskBox)
  const maskIndex = useAppSelector((state) => state.preview.maskIndex)
  // const maskVisible = useAppSelector((state) => state.controlPanel.maskVisible)
  const selectedMasks = useAppSelector((state) => state.preview.selectedMasks)
  const selectedMaskData = useAppSelector((state) => state.preview.selectedMaskData)
  const progress = useAppSelector((state) => state.preview.progress)
  const history = useAppSelector((state) => state.preview.history)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const status = useAppSelector((state) => state.preview.status)
  const layers = useAppSelector((state) => state.layerTable.layers)
  const [mask, setMask] = useState<HTMLImageElement | null>(null)
  const scaledSelectionBox = useAppSelector((state) => state.preview.scaledSelectionBox)

  const pointerDownRef = useRef<boolean>(false)
  const selectionBox = useRef([0, 0, 0, 0])

  const pointerCanvasCoords = useRef<number[]>([])
  const debounce = useRef(false)

  const dispatch = useAppDispatch()

  const [currentImage, setCurrentImage] = useState<HTMLImageElement | null>(null)
  const [masks, setMasks] = useState<{ [key: string]: HTMLImageElement }[]>([])
  const contextMenuOpen = useRef(false)
  const [contextMenuState, setContextMenuState] = useState<'open' | 'closed'>('closed')
  const [maskVisible, setMaskVisible] = useState(false)
  const [selectionBoxVisible, setSelectionBoxVisible] = useState(false)

  const drawCurrentImage = () => {
    if (!canvasRef.current) return
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

  const drawMasks = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    if (masks.length && maskVisible) {
      for (const mask of masks) {
        const index = masks.indexOf(mask)
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

        ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
      }
    }
  }

  const drawSelectionBox = () => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return

    const [x, y, w, h] = selectionBox.current

    if (selectionBoxVisible) {
      ctx.strokeStyle = 'white'
      ctx.setLineDash([15, 15])
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
    }
  }

  const pointerDown: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current || contextMenuOpen.current || !currentImage) return
    const canv = canvasRef.current as HTMLCanvasElement
    pointerDownRef.current = true
    const { x: bx, y: by } = canv.getBoundingClientRect()
    const x = e.clientX - bx
    const y = e.clientY - by
    pointerCanvasCoords.current = [x, y, 0, 0]
    selectionBox.current = [x, y, 0, 0]
    setSelectionBoxVisible(true)
  }

  const pointerMove: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement

    if (pointerDownRef.current && !contextMenuOpen.current) {
      const [x1, y1] = selectionBox.current
      const { x: bx, y: by } = canv.getBoundingClientRect()
      const x2 = e.clientX - bx
      const y2 = e.clientY - by
      const x = Math.min(x1, x2)
      const y = Math.min(y1, y2)
      const w = Math.abs(x2 - x1)
      const h = Math.abs(y2 - y1)
      selectionBox.current = [x, y, w, h]

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

        if (masks.length && maskVisible) {
          for (const mask of masks) {
            const index = masks.indexOf(mask)
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

            ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
          }
        }
      }
      if (selectionBoxVisible) {
        ctx.strokeStyle = 'white'
        ctx.setLineDash([15, 15])
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)
      }
    }
  }

  const pointerUp: PointerEventHandler<HTMLCanvasElement> = (e) => {
    if (!canvasRef.current || contextMenuOpen.current || !pointerDownRef.current) return
    console.log('pointerUp')
    dispatch(setMaskBox(selectionBox.current))
    const canv = canvasRef.current as HTMLCanvasElement
    pointerDownRef.current = false

    const [x1, y1] = selectionBox.current
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
      drawCurrentImage()
      drawMasks()
      drawSelectionBox()
      return
    }

    selectionBox.current = [x, y, w, h]
    dispatch(setMaskBox(selectionBox.current))

    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }

  const maskItemSelectHandler = (e) => {
    console.log('maskItemSelectHandler')
    e.stopPropagation()
    const [x, y, w, h] = maskBox
    console.log(`maskBox: ${maskBox}`)
    // if (w < 10 && h < 10) return

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

    const message = {
      image: history[currentHistoryIndex],
      bbox: bbox,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
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
          let new_data = [...data]
          new_data[0].active = true
          new_data = new_data.map((d) => ({ ...d, canvas_box: maskBox, bbox: bbox, active: false }))
          dispatch(setMaskData([...maskData, ...new_data].toSorted((d1, d2) => d2.area - d1.area)))
          dispatch(setMaskIndex(0))
          if (canvasRef.current) {
            drawCurrentImage()
            const canv = canvasRef.current as HTMLCanvasElement
            const ctx = canv.getContext('2d')
            if (!ctx) return
            const image = new Image()
            image.src = `data:image/png;base64,${new_data[0].segmentation}`
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

  const canvasWheelHandler: WheelEventHandler = (e) => {
    if (!canvasRef.current || !currentImage || debounce.current || !masks.length) return
    // e.preventDefault()
    e.stopPropagation()
    debounce.current = true

    const canv = canvasRef.current as HTMLCanvasElement
    console.log(`e.deltaY: ${e.deltaY}`)

    const [x1, y1] = selectionBox.current
    const { x: bx, y: by } = canv.getBoundingClientRect()
    const x2 = e.clientX - bx
    const y2 = e.clientY - by
    let x = Math.min(x1, x2)
    let y = Math.min(y1, y2)
    let w = Math.abs(x2 - x1)
    let h = Math.abs(y2 - y1)

    dispatch(
      setMaskData(
        maskData.map((data, index) => {
          const new_data = { ...data }
          if (data.id === maskData[maskIndex.value].id) {
            new_data.active = false
          }

          if (index === (maskIndex.value + 1) % maskData.length && e.deltaY > 0) {
            new_data.active = true
          }
          if (index === (maskIndex.value - 1) % maskData.length && e.deltaY <= 0) {
            new_data.active = true
          }
          return new_data
        })
      )
    )
    dispatch(
      setMaskIndex(
        e.deltaY > 0
          ? (maskIndex.value + 1) % maskData.length
          : (maskIndex.value - 1) % maskData.length
      )
    )
    selectionBox.current = [0, 0, 0, 0]

    const ctx = canv.getContext('2d')
    if (!ctx) return
    if (currentImage) {
      ctx.clearRect(x, y, w, h)
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
      ;[x, y, w, h] = maskData[0].canvas_box

      if (masks.length && maskVisible) {
        for (const mask of masks) {
          const index = masks.indexOf(mask)
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

          ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
        }
      }
    }
    setTimeout(() => (debounce.current = false), 500)
  }

  useEffect(() => {
    if (!canvasRef.current) return
    const canv = canvasRef.current as HTMLCanvasElement
    const ctx = canv.getContext('2d')
    if (!ctx) return
    const image = new Image()
    image.src = `data:image/png;base64,${history[currentHistoryIndex]}`
    image.onload = () => setCurrentImage(image)
  }, [currentHistoryIndex])

  useEffect(() => {
    if (!canvasRef.current || !currentImage || !masks) return
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

    if (masks.length && maskVisible) {
      console.log('masks: ', masks)
      for (const mask of masks) {
        const index = masks.indexOf(mask)
        console.log(`masks: `, masks)
        console.log(`mask index: ${index}`)
        const [x, y, w, h] = maskData[index].canvas_box

        let image = new Image()

        if (maskData[index].include) {
          image = mask.mask
        }
        if (maskData[index].exclude) {
          image = mask.inverted_mask
        }

        if (maskData[index].active) {
          image = mask.segmentation
        }
        image.onload = () => {
          ctx.drawImage(image, 0, 0, image.width, image.height, x, y, w, h)
        }
      }
    }

    const observer = new ResizeObserver(async (entries, target) => {
      const width = entries[0].borderBoxSize[0].inlineSize
      const height = entries[0].borderBoxSize[0].blockSize

      const size = Math.min(width, height)
      const scale_x = size / image.width
      const scale_y = size / image.height
      let scale = Math.min(scale_x, scale_y)

      const ctx = canv.getContext('2d')

      if (!ctx) return
      ctx.clearRect(0, 0, width, height)
      ctx.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        width / 2 - (image.width / 2) * scale,
        height / 2 - (image.height / 2) * scale,
        image.width * scale,
        image.height * scale
      )

      drawMasks()

      let [x, y, w, h] = selectionBox.current

      let scaled_x = Math.min(width, 1024) / currentImage.width
      let scaled_y = Math.min(height, 1024) / currentImage.height
      scale = Math.min(scaled_x, scaled_y)

      selectionBox.current = [x, y, w, h]
      dispatch(setMaskBox(selectionBox.current))

      drawSelectionBox()

      dispatch(
        setPreviewStatus({
          ...status,
          canvas: `width: ${canv.width}, height: ${canv.height}`,
          image: `width: ${image.width}, height: ${image.height}`,
          scale: scale,
          currentHistoryIndex: currentHistoryIndex,
        })
      )
    })

    // observer.observe(canv)

    return () => {
      observer.disconnect()
    }
  }, [currentImage])

  useEffect(() => {
    if (!maskData.length) return
    console.log(`maskData: `, maskData)
    const maskImages: { [key: string]: HTMLImageElement }[] = []

    for (const data of maskData) {
      console.log(`data: ${data}`)
      let keys: string[] = ['segmentation', 'mask', 'inverted_mask']
      const images: { [key: string]: HTMLImageElement } = {}
      for (const key of keys) {
        const b64 = data[key]
        const url = `data:image/png;base64,${b64}`
        const image = new Image()
        image.src = url
        image.onload = () => {
          images[key] = image
        }
      }
      maskImages.push(images)
    }
    setMasks(maskImages)
    if (canvasRef.current) {
      const canv = canvasRef.current as HTMLCanvasElement
      const ctx = canv.getContext('2d')
      if (!ctx) return
    }
  }, [maskData])

  useEffect(() => {
    if (
      !canvasRef.current ||
      history.length === 0 ||
      currentHistoryIndex >= history.length ||
      currentHistoryIndex < 0
    )
      return
    console.log('Preview updating...')
    const canv = canvasRef.current as HTMLCanvasElement
    const { width, height } = canv.getBoundingClientRect()
    canv.width = width
    canv.height = height
    const ctx = canv.getContext('2d')
    if (!ctx) return

    const image = new Image()
    image.src = `data:image/png;base64,${history[currentHistoryIndex]}`

    image.onload = () => {
      ctx.clearRect(0, 0, canv.width, canv.height)
      setCurrentImage(image)
    }
  }, [history])

  useEffect(() => {
    if (!masks.length) return
    console.log(`useEffect masks`)
    console.log(`masks: `, masks)
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }, [masks])

  useEffect(() => {
    if (!maskData.length) return
    console.log(`useEffect maskIndex`)
    drawCurrentImage()
    drawMasks()
    drawSelectionBox()
  }, [maskIndex])

  // const removeSelect = () => {
  //   dispatch(
  //     setSelectedMaskData(
  //       selectedMaskData.filter((data) => data.id !== maskData[maskIndex.value].id)
  //     )
  //   )
  // }

  return (
    <ResizablePanelGroup
      direction="vertical"
      className="relative flex flex-col items-center justify-start w-full h-full"
    >
      <ResizablePanel>
        <div className="flex h-[48px] min-h-[48px] items-start justify-between w-full bg-neutral-900 p-2">
          <div className="flex justify-start items-center gap-x-2 w-full">
            <button
              onClick={() => {
                console.log('Back button clicked')
                dispatch(setCurrentHistoryIndex(currentHistoryIndex - 1))
              }}
              disabled={currentHistoryIndex === 0}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                height="24px"
                viewBox="0 -960 960 960"
                width="24px"
                fill="#e8eaed"
              >
                <path d="m313-440 224 224-57 56-320-320 320-320 57 56-224 224h487v80H313Z" />
              </svg>
            </button>
            <button
              onClick={() => {
                console.log('Forward button clicked')
                dispatch(setCurrentHistoryIndex(currentHistoryIndex + 1))
              }}
              disabled={currentHistoryIndex === history.length - 1}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                height="24px"
                viewBox="0 -960 960 960"
                width="24px"
                fill="#e8eaed"
              >
                <path d="M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z" />
              </svg>
            </button>
            <p className="text-white text-sm self-justify-center">
              {currentHistoryIndex + 1}/{history.length}
            </p>
          </div>
          {/* <Progress value={progress} className=" top-0, left-0 w-full h-2" /> */}
        </div>
        <div className="flex flex-col relative justify-start w-full xl:min-h-[1024px]">
          <ContextMenu
            onOpenChange={(open) => {
              contextMenuOpen.current = open
              pointerDownRef.current = false
            }}
          >
            <ContextMenuTrigger>
              <canvas
                ref={canvasRef}
                className="relative w-full lg:h-fit cursor-crosshair"
                width={1024}
                height={1024}
                onPointerDown={pointerDown}
                onPointerMove={pointerMove}
                onPointerUp={pointerUp}
                // onWheel={canvasWheelHandler}
              ></canvas>
            </ContextMenuTrigger>
            <ContextMenuContent className="bg-neutral-800">
              <ContextMenuItem onSelect={(e) => maskItemSelectHandler(e)}>Mask</ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  setMaskVisible(!maskVisible)
                }}
              >
                {maskVisible ? 'Hide' : 'Show'}
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex.value) {
                          new_data.include = true
                          new_data.exclude = false
                        }
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, mask: imageData } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Include
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        if (index == maskIndex.value) {
                          new_data.include = false
                          new_data.exclude = true
                        }
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, inverted_mask: imageData } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Exclude
              </ContextMenuItem>
              <ContextMenuItem
                disabled={masks.length === 0}
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
                  let newSelectedMasks = [...selectedMasks]
                  const { id, segmentation: imageData } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Deselect
              </ContextMenuItem>
              <ContextMenuItem
                onSelect={(e) => {
                  e.stopPropagation()
                  dispatch(
                    setMaskData(
                      maskData.map((data, index) => {
                        const new_data = { ...data }
                        new_data.include = false
                        new_data.exclude = false
                        return new_data
                      })
                    )
                  )
                  let newSelectedMasks = [...selectedMasks]
                  const { id, segmentation: imageData } = maskData[maskIndex.value]
                  if (maskIndex.value > selectedMasks.length - 1) {
                    newSelectedMasks.push({ id, imageData })
                  } else {
                    newSelectedMasks = newSelectedMasks.toSpliced(
                      newSelectedMasks.indexOf(
                        newSelectedMasks.find((m) => m.id === id) as SelectedMask
                      ),
                      1,
                      { id, imageData }
                    )
                  }
                  dispatch(setSelectedMasks(newSelectedMasks))
                }}
              >
                Deselect All
              </ContextMenuItem>
              <ContextMenuItem
                onSelect={() => {
                  dispatch(setSelectedMasks([]))
                  setMasks([])
                  dispatch(setMaskData([]))
                  drawCurrentImage()
                }}
              >
                Clear
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
          <div className="h-[2rem] w-full bg-neutral-800 text-white text-sm p-2 m-0 flex gap-x-4 overflow-clip">
            {Object.entries(status).map((entry) => (
              <span key={entry[0]}>{`${entry[0]}: ${entry[1]}`}</span>
            ))}
          </div>
        </div>
      </ResizablePanel>
      <ResizableHandle className="border-solid border-neutral-800 border-2" />
      <ResizablePanel className="w-full">
        <div className="flex items-center justify-center space-x-4 w-full">
          <ToolPalette />
          {/*<Button
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
                    const new_data = { ...data }
                    if (index === maskIndex.value) {
                      new_data.active = false
                      new_data.include = true
                      new_data.exclude = false
                    }
                    return new_data
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
          </Button>*/}
          {/* <Button size="lg" aria-label="remove select" onClick={removeSelect}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="48px"
              viewBox="0 -960 960 960"
              width="48px"
              fill="#e8eaed"
            >
              <path d="m500-120-56-56 142-142-142-142 56-56 142 142 142-142 56 56-142 142 142 142-56 56-142-142-142 142Zm-220 0v-80h80v80h-80Zm-80-640h-80q0-33 23.5-56.5T200-840v80Zm80 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80q33 0 56.5 23.5T840-760h-80ZM200-200v80q-33 0-56.5-23.5T120-200h80Zm-80-80v-80h80v80h-80Zm0-160v-80h80v80h-80Zm0-160v-80h80v80h-80Zm640 0v-80h80v80h-80Z" />
            </svg>
          </Button> */}
        </div>
        <PromptPanel />
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
