import Pynbtemplate from "./components/PynbTemplate";
import VuePrism from "vue-prism";

const PynbtemplateSimple = {
    install(Vue) {
        Vue.use(VuePrism)
        Vue.component("pynb-template", Pynbtemplate);
    }
};

// Automatic installation if Vue has been added to the global scope.
if (typeof window !== 'undefined' && window.Vue) {
    window.Vue.use(PynbtemplateSimple);
}

export default PynbtemplateSimple;
