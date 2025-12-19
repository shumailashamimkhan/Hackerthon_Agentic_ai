module.exports = {
  extends: [
    '@docusaurus/eslint-plugin',
    '@typescript-eslint/recommended',
    'prettier',
    'plugin:prettier/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: [
    '@typescript-eslint',
    'prettier',
  ],
  rules: {
    'prettier/prettier': 'error',
  },
  overrides: [
    {
      files: ['*.js', '*.ts', '*.tsx'],
      rules: {
        // Specific rules for JS/TS files
      },
    },
    {
      files: ['*.md', '*.mdx'],
      rules: {
        // Specific rules for markdown files if needed
      },
    },
  ],
};