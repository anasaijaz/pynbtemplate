<template>
  <div class="container">
    <p v-if="execution_count" class="line_in">In [{{ execution_count}}]:</p>
    <p v-else style="margin-right: 3.5rem"></p>

    <div v-if="cell['cell_type']==='code'" class="block">
    <pre class="language-js"><code  :style="code_style">{{src.join('')}}}</code>
</pre>

    </div>
    <div v-if="cell['cell_type']==='markdown'" v-html="renderMarkdown(src.join(''))">
    </div>

  </div>
</template>

<script>
import {marked} from 'marked';
import 'prismjs/themes/prism.css'

export default {
  name: "CellSource",
  props: ['src', 'line_in', 'execution_count','cell', 'code_font_size'],
  methods: {
    renderMarkdown(markdown) {
      return marked(markdown);
    }
  },
  computed: {
    code_style () {
      return `font-size: ${this.code_font_size} !important`
    }
  },
  components: {
  }
}
</script>

<style scoped>

.block {
  background-color: rgb(245, 245, 245);
  padding: 0.9rem;
  border: 1px solid rgb(224, 224, 224);
  border-radius: 0.25rem;
  flex-grow: 1;
  overflow: auto;
}
.line {
  margin: unset;
  line-height: 1.5;
  font-family: Consolas,serif;
}

.line_in {
  font-family: Consolas, monospace;
  margin: unset;
  font-size: small;
}

.container {
  display: flex;
  gap: 1.1rem;
}


</style>
