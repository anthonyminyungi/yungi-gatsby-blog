import { useEffect } from 'react'

export const useOnClickOutside = (ref, handler, events) => {
  if (!events) events = [`mousedown`, `touchstart`]
  useEffect(() => {
    const detectClickOutside = event =>
      !ref.current.contains(event.target) && handler()
    for (const event of events)
      document.addEventListener(event, detectClickOutside)
    return () => {
      for (const event of events)
        document.removeEventListener(event, detectClickOutside)
    }
  }, [ref, handler, events])
}
