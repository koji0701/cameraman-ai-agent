module.exports = {
  content: ['./src/renderer/**/*.{js,jsx,ts,tsx}', './src/components/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'accent-purple': 'var(--accent-purple)',
        'accent-teal': 'var(--accent-teal)',
        'accent-pink': 'var(--accent-pink)',
        'glass-white': 'rgba(255, 255, 255, var(--tw-bg-opacity, 1))',
      },
      backdropBlur: {
        xs: '2px',
        '4xl': '72px',
      },
      boxShadow: {
        'glass-sm': 'var(--glass-shadow-small)',
        'glass-md': 'var(--glass-shadow-medium)',
        'glass-lg': 'var(--glass-shadow-large)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
        'glow-teal': '0 0 20px rgba(20, 184, 166, 0.3)',
        'glow-pink': '0 0 20px rgba(236, 72, 153, 0.3)',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'breathe': 'breathe 4s ease-in-out infinite',
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'slide-in-up': 'slideInUp 0.8s ease-out forwards',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0) rotate(0deg)' },
          '25%': { transform: 'translateY(-10px) rotate(1deg)' },
          '50%': { transform: 'translateY(-20px) rotate(0deg)' },
          '75%': { transform: 'translateY(-10px) rotate(-1deg)' },
        },
        breathe: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.7' },
          '50%': { transform: 'scale(1.05)', opacity: '1' },
        },
        fadeInUp: {
          'from': {
            opacity: '0',
            transform: 'translateY(30px)',
          },
          'to': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        slideInUp: {
          'from': {
            opacity: '0',
            transform: 'translateY(100px)',
          },
          'to': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
      },
    },
  },
  variants: {
    extend: {
      backdropBlur: ['hover', 'focus'],
      backdropFilter: ['hover', 'focus'],
    },
  },
  plugins: [],
};