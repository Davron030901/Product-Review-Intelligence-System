import '@testing-library/jest-dom/vitest'

// jsdom implements neither of these, and Recharts + the app both need them.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver

if (!('randomUUID' in crypto)) {
  Object.defineProperty(crypto, 'randomUUID', {
    value: () => `test-${Math.random().toString(36).slice(2)}`,
  })
}
