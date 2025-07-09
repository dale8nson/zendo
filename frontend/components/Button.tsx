import { MouseEventHandler, useEffect, useRef, useState } from 'react'

export function Button({
  children,
  onClick,
  disabled,
  ...props
}: {
  children: React.ReactNode
  onClick: MouseEventHandler<HTMLButtonElement>
  disabled?: boolean
  props?: React.ComponentPropsWithoutRef<'button'>
}) {
  const [buttonPressed, setButtonPressed] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!ref.current) return
    const btn = ref.current as HTMLButtonElement
    const classes = btn.classList
    classes.toggle('hover:from-black')
    classes.toggle('hover:to-gray-900')
  }, [buttonPressed])

  return (
    <button
      ref={ref}
      className="w-full py-2 px-4 rounded-md text-white bg-gradient-to-br from-gray-800 to-black hover:from-black hover:to-gray-900 cursor-pointer  disabled:from-gray-500 disabled:to-gray-800 disabled:text-neutral-800/80"
      onPointerDown={() => setButtonPressed(true)}
      onPointerUp={(e) => {
        setButtonPressed(false)
        onClick(e)
      }}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  )
}
