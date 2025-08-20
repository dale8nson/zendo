'use client'

import { useEffect } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import {
  setGeneratePrompt,
  setGeneratePrompt2,
  setGenerateNegativePrompt,
  setGenerateNegativePrompt2,
  setImg2imgPrompt,
  setImg2imgPrompt2,
  setImg2imgNegativePrompt,
  setImg2imgNegativePrompt2,
  setInpaintPrompt,
  setInpaintPrompt2,
  setInpaintNegativePrompt,
  setInpaintNegativePrompt2,
  setDefaultPromptTab,
  setSelectedTab,
} from '@/lib/features/prompt-panel/promptPanelSlice'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { queryOptions, useQuery } from '@tanstack/react-query'

export function PromptPanel() {
  const generatePrompt = useAppSelector((state) => state.promptPanel.generate_prompt)
  const generateNegativePrompt = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt
  )
  const generatePrompt2 = useAppSelector((state) => state.promptPanel.generate_prompt2)
  const generateNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt2
  )
  const img2imgPrompt = useAppSelector((state) => state.promptPanel.img2img_prompt)
  const img2imgNegativePrompt = useAppSelector((state) => state.promptPanel.img2img_negativePrompt)
  const img2imgPrompt2 = useAppSelector((state) => state.promptPanel.img2img_prompt2)
  const img2imgNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.img2img_negativePrompt2
  )
  const inpaintPrompt = useAppSelector((state) => state.promptPanel.inpaint_prompt)
  const inpaintNegativePrompt = useAppSelector((state) => state.promptPanel.inpaint_negativePrompt)
  const inpaintPrompt2 = useAppSelector((state) => state.promptPanel.inpaint_prompt2)
  const inpaintNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.inpaint_negativePrompt2
  )
  const defaultPromptTab = useAppSelector((state) => state.promptPanel.default_prompt_tab)
  const selectedTab = useAppSelector((state) => state.promptPanel.selected_tab)
  const dispatch = useAppDispatch()

  const {
    data: tokenList,
    error,
    isLoading: tokensLoading,
    isError,
    refetch,
  } = useQuery(
    queryOptions({
      queryKey: ['tokens'],
      queryFn: async () => {
        const res = await fetch(`http://127.0.0.1:8000/api/tokens`)
        if (!res.ok) {
          throw new Error(`Failed to fetch tokens: ${res.status} ${res.statusText}`)
        }
        return await res.json()
      },
      refetchOnWindowFocus: false,
      staleTime: Infinity,
    })
  )

  useEffect(() => {
    dispatch(setGeneratePrompt(localStorage.getItem('generate_prompt') || ''))
    dispatch(setGenerateNegativePrompt(localStorage.getItem('generate_negative_prompt') || ''))
    dispatch(setGeneratePrompt2(localStorage.getItem('generate_prompt_2') || ''))
    dispatch(setGenerateNegativePrompt2(localStorage.getItem('generate_negative_prompt_2') || ''))
    dispatch(setImg2imgPrompt(localStorage.getItem('img2img_prompt') || ''))
    dispatch(setImg2imgNegativePrompt(localStorage.getItem('img2img_negative_prompt') || ''))
    dispatch(setImg2imgPrompt2(localStorage.getItem('img2img_prompt_2') || ''))
    dispatch(setImg2imgNegativePrompt2(localStorage.getItem('img2img_negative_prompt_2') || ''))
    dispatch(setInpaintPrompt(localStorage.getItem('inpaint_prompt') || ''))
    dispatch(setInpaintNegativePrompt(localStorage.getItem('inpaint_negative_prompt') || ''))
    dispatch(setInpaintPrompt2(localStorage.getItem('inpaint_prompt_2') || ''))
    dispatch(setInpaintNegativePrompt2(localStorage.getItem('inpaint_negative_prompt_2') || ''))
    dispatch(setDefaultPromptTab(localStorage.getItem('default_prompt_tab') || 'generate'))
  }, [])

  return (
    <Tabs className="flex flex-col w-full h-full m-0" defaultValue={defaultPromptTab}>
      <TabsList className="flex w-full items-center space-x-4">
        <TabsTrigger className="font-bold text-lg" value="generate">
          Generate
        </TabsTrigger>
        <TabsTrigger className="font-bold text-lg" value="img2img">
          Img2Img
        </TabsTrigger>
        <TabsTrigger className="font-bold text-lg" value="inpaint">
          Inpaint
        </TabsTrigger>
      </TabsList>
      <TabsContent className="w-full h-full m-0" value="generate">
        <div className="grid grid-cols-2 gap-2 w-full h-full px-2 m-0">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="generate_prompt"
              value={generatePrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setGeneratePrompt(value))
                  localStorage.setItem('generate_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white !w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="negative_prompt"
              value={generateNegativePrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setGenerateNegativePrompt(value))
                  localStorage.setItem('generate_negative_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="generate_prompt_2"
              value={generatePrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setGeneratePrompt2(value))
                  localStorage.setItem('generate_prompt_2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="generate_negative_prompt_2"
              value={generateNegativePrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setGenerateNegativePrompt2(value))
                  localStorage.setItem('generate_negative_prompt2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
      <TabsContent value="img2img">
        <div className="grid grid-cols-2 gap-2 space-x-2 justify-start w-full h-full px-2">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="img2img_prompt"
              value={img2imgPrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setImg2imgPrompt(value))
                  localStorage.setItem('img2img_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="img2img_negative_prompt"
              value={img2imgNegativePrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setImg2imgNegativePrompt(value))
                  localStorage.setItem('img2img_negative_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>

          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="img2img_prompt_2"
              value={img2imgPrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setImg2imgPrompt2(value))
                  localStorage.setItem('img2img_prompt_2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="img2img_negative_prompt_2"
              value={img2imgNegativePrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setImg2imgNegativePrompt2(value))
                  localStorage.setItem('img2img_negative_prompt_2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
      <TabsContent value="inpaint">
        <div className="grid grid-cols-2 gap-2 space-x-2 justify-start w-full h-full px-2">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="inpaint_prompt"
              value={inpaintPrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setInpaintPrompt(value))
                  localStorage.setItem('inpaint_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="inpaint_negative_prompt"
              value={inpaintNegativePrompt}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setInpaintNegativePrompt(value))
                  localStorage.setItem('inpaint_negative_prompt', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="inpaint_prompt_2"
              value={inpaintPrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setInpaintPrompt2(value))
                  localStorage.setItem('inpaint_prompt_2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="inpaint_negative_prompt_2"
              value={inpaintNegativePrompt2}
              onChange={(e) => {
                if (!tokensLoading) {
                  const tokens = tokenList.tokens

                  console.log(`tokens: `, tokens)
                  let value = e.target.value
                  for (const tk of tokens) {
                    console.log(`tk: ${tk}`)
                    console.log(`tk.slice(1, tk.length - 1): ${tk.slice(1, tk.length - 1)}`)
                    // value = e.target.value.replaceAll(tk.slice(1, tk.length - 1), ` ${tk}`)

                    value = value.replace(
                      new RegExp(`[^<\\W]?(${tk.slice(1, tk.length - 1)})[^>]`),
                      '<$1>'
                    )
                  }
                  console.log(`value: ${value}`)
                  e.target.value = value
                  dispatch(setInpaintNegativePrompt2(value))
                  localStorage.setItem('inpaint_negative_prompt_2', value)
                }
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
    </Tabs>
  )
}
