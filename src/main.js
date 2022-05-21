import Vue from 'vue'
import App from './App.vue'
import VuePrism from "vue-prism";

Vue.config.productionTip = false
Vue.use(VuePrism)

new Vue({
  render: h => h(App),
}).$mount('#app')
