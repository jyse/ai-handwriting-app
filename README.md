## 🎨 How `bg-bg` Works – Theming Explained

Tailwind lets you write classnames like `bg-bg`, `text-primary`, etc. These are powered by:

- `tailwind.config.mjs` defines token names:

  ```js
  colors: {
    bg: 'var(--color-bg)',
    text: 'var(--color-text)',
    primary: 'var(--color-primary)',
  }
  ```

- `globals.css` defines the actual values for these tokens:

  ```css
  :root {
    --color-bg: #f7f7f7;
    --color-text: #331e38;
    ...;
  }

  html.dark {
    --color-bg: #191923;
    --color-text: #99907d;
    ...;
  }
  ```

### 🧠 Why This Is Great:

- Your components stay **clean and semantic** (`bg-bg`, not `bg-[#191923]`)
- You can **theme your app globally** just by changing the `:root` CSS vars
- Dark/light mode is handled automatically with `html.dark`

### 🪄 Visual Flow:

```
bg-bg (in component)
    ↓
Tailwind token `bg` → 'var(--color-bg)'
    ↓
CSS var `--color-bg` → changes based on light/dark mode
    ↓
Final result: background color updates instantly across the app
```
