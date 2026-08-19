/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#FDFCF8',
        secondary: '#F7F3EA',
        card: '#FFFEFA',
        accent: {
          primary: '#176B4D',
          dark: '#0F5138',
          light: '#E8F1E9',
          subtle: '#DDECE2',
        },
        neutral: {
          warm: '#E9DDC8',
          soft: '#F2EBDD',
          border: '#E5DED1',
        },
        text: {
          primary: '#1E2A24',
          secondary: '#68736D',
        }
      },
      fontFamily: {
        sans: ['Inter', 'DM Sans', 'sans-serif'],
        serif: ['Playfair Display', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'monospace']
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
      }
    },
  },
  plugins: [],
}
