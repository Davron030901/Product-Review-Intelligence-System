/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Base: cool pale paper, not stark white, not cream.
        paper: '#ECEEEA',
        card: '#F6F7F4',
        ink: '#161A17',
        slate: '#4A5A52',
        line: '#D2D7CF',
        // Primary: deep pine. Actions, branding, the left rail.
        pine: { DEFAULT: '#23483C', deep: '#182F27', soft: '#3A6555' },
        // Signal amber. Reserved exclusively for "needs review". Nothing
        // else in the app is allowed to use it, so the eye learns it.
        signal: { DEFAULT: '#B87611', ink: '#7A4E08', wash: '#F5E7CC' },
        // One hue per issue category, used identically everywhere.
        cat: {
          delivery: '#2F6F8F',
          packaging: '#7A5C9E',
          quality: '#2E7D5B',
          defect: '#B03A3A',
          price: '#8A6414',
          service: '#B04A7D',
          fit: '#5B7A2E',
          other: '#5F655D',
        },
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['"Public Sans"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(22,26,23,0.06), 0 8px 24px -16px rgba(22,26,23,0.30)',
        rail: '1px 0 0 rgba(22,26,23,0.08)',
      },
      keyframes: {
        'stamp-in': {
          '0%': { opacity: '0', transform: 'rotate(-14deg) scale(1.6)' },
          '60%': { opacity: '1', transform: 'rotate(-4deg) scale(0.94)' },
          '100%': { opacity: '1', transform: 'rotate(-6deg) scale(1)' },
        },
        'rise': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'stamp-in': 'stamp-in 420ms cubic-bezier(0.2, 0.9, 0.3, 1) both',
        'rise': 'rise 260ms ease-out both',
      },
    },
  },
  plugins: [],
}
