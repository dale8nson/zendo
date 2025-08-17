'use clinet'

import { useState, useRef } from 'react'
import { RgbaColorPicker, HexAlphaColorPicker, HexColorInput } from 'react-colorful'
import { Button } from './Button'
import { PopoverPicker } from './PopoverPicker'

export function ToolPalette() {
  const [color, setColor] = useState('#ffffffff')
  const debounce = useRef(false)

  const fillMask = () => {}

  const textChange = (e) => {
    if (debounce.current) return
    debounce.current = true
    setColor(e.target.value)
    setTimeout(() => (debounce.current = false), 500)
  }

  return (
    <div className="flex items-center justify-center w-full">
      <div className="flex-col items-center ">
        {/*<PopoverPicker color={color} onChange={setColor} />
        <input type="text" value={color} onChange={(e) => textChange} />*/}
        <Button onClick={fillMask}>Fill Mask</Button>
      </div>
    </div>
  )
}
