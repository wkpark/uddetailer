function mudd_inpainting() {
    var res = Array.from(arguments);

    // get the current selected gallery id
    var idx = selected_gallery_index();
    res[2] = idx; // gallery id

    return submit.apply(this, res);
}

function mudd_inpainting_img2img() {
    var res = Array.from(arguments);

    // get current tabId
    var tab = get_img2img_tab_index();
    var tabid = tab[0]; // tab Id

    var tabs = [ "#img2img_img2img_tab", "#img2img_img2img_sketch_tab", "#img2img_inpaint_tab", "#img2img_inpaint_sketch_tab", "#img2img_inpaint_upload_tab" ];
    if (tabid > 4 || tabid < 0)
        tabid = 0;

    // get base64 src
    var src = gradioApp().querySelector(tabs[tabid]).querySelectorAll("img")[0].src;

    res[3] = src; // gr.Image() component
    res[1] = tab[0]; // tab Id

    var idx = selected_gallery_index();
    res[2] = idx; // gallery Id

    return submit_img2img.apply(this, res);
}

function get_gallery_id(tabname) {
    var gallery_id =  "mudd_inpainting_image";
    if (tabname == "img2img") {
        // get the current tabId
        var tab = get_img2img_tab_index();
        var tabid = tab[0]; // tab Id

        var tabs = [ "img2img_image", "img2img_sketch", "img2maskimg", "inpaint_sketch" ];
        if (tabid > 4 || tabid < 0)
            tabid = 0;
        gallery_id = tabs[tabid];
    }

    return gallery_id
}

function overlay_masks() {
    var res = Array.from(arguments);

    var label, is_img2img, selected, sync;
    var gallery_mode = false;

    if (res.length == 3) {
        gallery_mode = true;
        label = res[1];
        is_img2img = res[2];
        selects = res[0];
        selected = '';
        sync = true;
    } else {
        label = res[3];
        is_img2img = res[4];

        selects = res[0];
        selected = res[1];
        var options = res[2];

        sync = false;
        if (typeof options == "boolean") {
            sync = options;
        } else {
            if (options.indexOf("sync") != -1) {
                sync = true;
            }
        }
    }

    var selects = selects.join(",");
    if (sync) {
        selected += "," + selects;
    } else {
        selected = selects;
    }
    var parts = selected.split(",");
    // strip possible label prefix. e.g) "A-face 0.92:3" -> 3
    var sels = [];
    for (var i = 0; i < parts.length; i++) {
        var j, tmp;
        if ((j = parts[i].lastIndexOf(":")) != -1 && parts[i].startsWith(label)) {
            tmp = parts[i].substring(j+1);
        } else {
            tmp = parts[i];
        }

        if ((j = tmp.indexOf("-")) != -1) {
            // expand range e.g) 3-5 -> 3,4,5
            var dummy = tmp.split("-");
            if (dummy.length == 2) {
                var start = parseInt(dummy[0]);
                var end = parseInt(dummy[1]);
                if (isNaN(start) || isNaN(end)) {
                    // ignore
                    continue;
                }
                if (start == end) {
                    sels.push(start);
                    continue;
                }
                if (end < start) {
                    var x = end;
                    end = start;
                    start = x;
                }
                var append = [...Array(end - start + 1).keys()];
                for (var k = 0; k < append.length; k++) {
                    sels.push(append[k] + start);
                }
            }
        } else {
            var num = parseInt(tmp);
            if (!isNaN(num) && num > 0) {
                sels.push(num);
            }
        }
    }

    sels = sels.map((v) => v - 1);

    var tabname = is_img2img ? "img2img":"txt2img";
    var gallery_id = tabname + "_gallery";
    if (!gallery_mode) {
        gallery_id = get_gallery_id(tabname);
        var gallery = gradioApp().querySelector("#" + gallery_id + ' .image-container');
        var imgs = gallery.querySelectorAll("img")
        if (imgs.length == 0) {
            gallery_id = tabname + "_gallery";
        }
    }

    var masks_id = "#mudd_masks_" + label.toLowerCase() + "_" + tabname;
    if (gallery_id.indexOf("_gallery") != -1) {
        // inpainting helper has no image
        masks_id = "#mudd_masks_" + label.toLowerCase() + "_gallery_" + tabname;
    }
    var masks_data = gradioApp().querySelector(masks_id + " textarea");
    var masks;
    if (masks_data.value) {
        try {
            masks = JSON.parse(masks_data.value);
        } catch(e) {
            // only one detection found case
            var bbox = masks_data.value.split(",");
            if (bbox.length == 5) {
                var lab = bbox.splice(0, 1);
                lab = lab[0].split(" ");
                var score = lab.splice(-1)
                masks = { "labels": [lab.join(" ")], "scores": [parseFloat(score)] , "bboxes": [ bbox.map(x => parseInt(x)) ] }
            }
        }
    }

    // check lightboxModal and copy mask canvas into lightboxModal as an image
    var lightbox = gradioApp().getElementById("lightboxModal");
    var lightbox_wrap = lightbox.querySelector(".mudd_masks_wrapper");
    if (!lightbox_wrap) {
        lightbox_wrap = document.createElement("div");
        lightbox_wrap.className = "mudd_masks_wrapper";
        lightbox_wrap.style.display = "flex";
        lightbox_wrap.style.justifyContent = "center";

        // misc fixes to fit in wrapper
        lightbox_wrap.style.height = "100%";
        lightbox_wrap.style.width = "100%";
        lightbox_wrap.style.position = "absolute";
        lightbox_wrap.style.top = '0px';

        var img = lightbox.querySelector("img");
        if (img) {
            lightbox.insertBefore(lightbox_wrap, img);
        }
    }

    var canvas = make_mask(masks, sels, is_img2img, gallery_mode);
    if (canvas) {
        console.log("canvas created");
    }

    if (gallery_id.indexOf("_gallery") != -1 && lightbox_wrap && canvas) {
        console.log("copy canvas to img")
        var imageData = canvas.toDataURL("image/png");
        if (lightbox_wrap.firstChild) {
            lightbox_wrap.removeChild(lightbox_wrap.firstChild);
        }
        var img = document.createElement("img");
        img.naturalWidth = canvas.width;
        img.naturalHeight = canvas.height;
        img.src = imageData;
        lightbox_wrap.appendChild(img);
    } else if (lightbox_wrap?.firstChild) {
        lightbox_wrap.removeChild(lightbox_wrap.firstChild);
    }

    return res;
}

function reset_masks() {
    var res = Array.from(arguments);

    var is_img2img = res[2];

    var tabname = is_img2img ? "img2img" : "txt2img";

    var gallery_id = get_gallery_id(tabname);
    var gallery = gradioApp().querySelector("#" + gallery_id + ' .image-container');
    var imgs = gallery.querySelectorAll("img")
    if (imgs.length == 0) {
        // get canvas wrapper
        var wrap = gradioApp().querySelectorAll("#" + gallery_id + " .mudd_masks_wrapper")[0];
        if (wrap) {
            gallery.removeChild(wrap);
        }
        gallery_id = tabname + "_gallery";
        console.log("XXX mask reset inpaint image")
    }

    if (gallery_id.indexOf("_gallery") != -1) {
        console.log("XXX mask reset gallery")
        // check for gallery
        gallery = gradioApp().querySelector('#' + gallery_id);
        imgs = gallery.querySelectorAll(".preview > img");
        if (imgs.length == 0) {
            wrap = gradioApp().querySelectorAll("#" + gallery_id + " .mudd_masks_wrapper")[0];
            if (wrap) {
                gallery.removeChild(wrap);
            }

            // check for modal
            var lightbox = gradioApp().getElementById("lightboxModal");
            var lightbox_wrap = lightbox.querySelector(".mudd_masks_wrapper");
            if (lightbox_wrap) {
                lightbox.removeChild(lightbox_wrap);
            }
        }
    }

    // gradio bug XXX
    res[0] = res[0] || gradioApp().querySelector("#mudd_masks_" + res[1].toLowerCase() + "_" + tabname + " textarea").value;

    return res;
}

function gallery_get_masks() {
    var res = Array.from(arguments);

    var is_img2img = res[2];

    var tabname = is_img2img ? "img2img" : "txt2img";
    var gallery_id = tabname + "_gallery";

    res[0] = gradioApp().querySelector("#mudd_masks_" + res[1].toLowerCase() + "_gallery_" + tabname + " textarea").value;

    return res;
}

function make_mask(masks, selected, is_img2img, is_gallery) {
    var segms = masks?.segms;
    var bboxes = masks?.bboxes;
    var labels = masks?.labels;
    var scores = masks?.scores;

    var tabname = is_img2img ? "img2img" : "txt2img";

    var gallery_id = tabname + "_gallery";
    var gallery;
    var imgs;
    if (!is_gallery) {
        gallery_id = get_gallery_id(tabname);
        gallery = gradioApp().querySelector("#" + gallery_id + ' .image-container');
        imgs = gallery.querySelectorAll("img")
    } else {
        gallery = gradioApp().querySelector('#' + gallery_id);
        imgs = gallery.querySelectorAll(".preview > img");
    }

    // check wrapper
    var wrap = gradioApp().querySelectorAll("#" + gallery_id + " .mudd_masks_wrapper")[0];
    if (!wrap) {
        wrap = document.createElement("div");
        wrap.className = "mudd_masks_wrapper";
        wrap.style.display = "flex";
        wrap.style.justifyContent = "center";

        // for gr.Image()
        wrap.style.height = "100%";
        wrap.style.width = "100%";
        wrap.style.position = "absolute";
        wrap.style.top = '0px';

        gallery.appendChild(wrap);
    }

    var canvas = wrap.getElementsByTagName("canvas")[0];
    if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.style.position = "absolute";
        canvas.style.top = '0px';
        canvas.style.zIndex = '100';
        canvas.className = "mask-overlay";
        canvas.style.pointerEvents = "none";

        wrap.appendChild(canvas);
    }

    if (!masks) {
        // no masks info. clear canvas
        var ctx = canvas.getContext('2d');
        if (imgs.length > 0) {
            var img = imgs[0];
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            canvas.style.height = img.style.height || "100%";
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return canvas;
    }

    if (imgs.length > 0) {
        // from https://stackoverflow.com/a/47593316/1696120
        function sfc32(a, b, c, d) {
            return function() {
                a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
                var t = (a + b) | 0;
                a = b ^ b >>> 9;
                b = c + (c << 3) | 0;
                c = (c << 21 | c >>> 11);
                d = d + 1 | 0;
                t = t + d | 0;
                c = c + t | 0;
                return (t >>> 0) / 4294967296;
            }
        }

        var random = random = sfc32(0x9E3779B9, 0x243F6A88, 0xB7E15162, 1377);

        function srand(seed) {
            var seed = seed ^ 0xDEADBEEF; // 32-bit seed with optional XOR value
            // Pad seed with Phi, Pi and E.
            // https://en.wikipedia.org/wiki/Nothing-up-my-sleeve_number
            random = sfc32(0x9E3779B9, 0x243F6A88, 0xB7E15162, seed);
        }

        function randint(min, max) {
            return Math.floor(random() * (max - min + 1)) + min;
        }

        var img = imgs[0];

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.height = img.style.height || "100%";

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.font = "25px sans-serif";
        for (var i = 0; i < bboxes?.length; i++) {
            // always generate colors to fix the order of colors.
            var red = randint(100, 255);
            var green = randint(100, 255);
            var blue = randint(100, 255);

            if (selected.indexOf(i) == -1) {
                continue;
            }

            if (segms) {
                ctx.fillStyle = 'rgba(' + red + ',' + green + ',' + blue + ',0.3)';
            } else {
                ctx.fillStyle = 'rgba(' + red + ',' + green + ',' + blue + ',0.6)';
            }
            var label = labels[i];
            var score = scores[i];
            if (score > 0) {
                label += " " + score.toFixed(2);
            }
            label += ":" + (i+1);
            var sz = ctx.measureText(label);
            var bbox = bboxes[i];
            ctx.fillRect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]);

            ctx.fillStyle = 'rgba(' + red + ',' + green + ',' + blue + ',0.6)';
            if (segms) {
                var poly = segms[i];
                for (var k = 0; k < poly.length; k++) {
                    var seg = poly[k];
                    if (typeof seg[0] == 'number') {
                        // 1-dim array to 2dim array: x,y,x1,y1,x2,y2 -> [x,y], [x1,y1], [x2,y2], ...
                        var tmp = [];
                        while(seg.length)
                            tmp.push(seg.splice(0,2));
                        seg = tmp;
                    }

                    ctx.beginPath();
                    ctx.moveTo(seg[0][0], seg[0][1]);
                    for (var j = 1; j < seg.length; j++) {
                        ctx.lineTo(seg[j][0], seg[j][1]);
                    }
                    ctx.closePath();
                    ctx.fill();
                }
            }

            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, (bbox[0] + bbox[2])/2 - sz.width/2, (bbox[1] + bbox[3])/2);
        }

        return canvas;
    }
    return null;
}

(function() {
let generation_buttons = {};
let enabled = {};

function parse_generation_info(tabname, infotext) {
    let extra_generation_params = infotext.split("Negative prompt:")[1].split("\n").slice(-1)[0];
    let matches = [...extra_generation_params.matchAll(/\s*(\w[\w \-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/g)]
    let params = {};
    for (let i = 0; i < matches.length; i++) {
        if (matches[i][2][0] == '"' && matches[i][2].slice(-1) == '"') {
            params[matches[i][1]] = JSON.parse(matches[i][2]);
        } else {
            params[matches[i][1]] = matches[i][2];
        }
    }
    let masks_a = gradioApp().querySelector("#mudd_masks_a_gallery_" + tabname + " textarea");
    let masks_b = gradioApp().querySelector("#mudd_masks_b_gallery_" + tabname + " textarea");
    if (params["MuDDetailer detection a"] && masks_a) {
        if (masks_a.value != params["MuDDetailer detection a"]) {
            masks_a.value = params["MuDDetailer detection a"];
            // update selectable detected masks dropdown
            gradioApp().querySelector("#mudd_masks_a_change_" + tabname).click();
        }
    } else if (!params["MuDDetailer detection a"]) {
        masks_a.value = '';
        gradioApp().querySelector("#mudd_masks_a_change_" + tabname).click();
    }
    if (params["MuDDetailer detection b"] && masks_b) {
        if (masks_b.value != params["MuDDetailer detection b"]) {
            masks_b.value = params["MuDDetailer detection b"];
            gradioApp().querySelector("#mudd_masks_b_change_" + tabname).click();
        }
    } else if (!params["MuDDetailer detection b"]) {
        masks_b.value = '';
        gradioApp().querySelector("#mudd_masks_b_change_" + tabname).click();
    }
}

function attachGalleryButtonListener(tabname) {
    gradioApp().getElementById(tabname + "_generation_info_button")?.addEventListener('click', (e) => {
        if (!opts.mudd_use_gallery_detection_preview)
            return;

        let id = selected_gallery_index();
        let src = gradioApp().querySelector("#generation_info_" + tabname + " textarea").value;
        if (id >= 0 && src) {
            let infotexts = JSON.parse(src).infotexts;
            if (infotexts[id]) {
                parse_generation_info(tabname, infotexts[id]);
            }
        }

    });
}

// attach mask button at the bottom of gallery
function attach_mask_button() {
    if (!opts.mudd_use_gallery_detection_preview)
        return;
    var gallery = get_current_gallery();
    var tabname = gallery?.id.replace("_gallery", "");
    if (!tabname)
        return;

    var buttons = gradioApp().querySelectorAll("#image_buttons_" + tabname + " .form button");
    if (buttons.length > 0 && buttons[buttons.length - 1].id != "") {
        var maskbtn = document.createElement("button");
        maskbtn.className = buttons[buttons.length - 1].classList;
        maskbtn.title = "Hide/Show masks";
        maskbtn.innerHTML="\u{1f3ad}";
        maskbtn.addEventListener('click', (e) => {
            if (!opts.mudd_use_gallery_detection_preview)
                return;

            let masks_a = gradioApp().querySelector("#mudd_masks_a_gallery_" + tabname + " textarea");
            let masks_b = gradioApp().querySelector("#mudd_masks_b_gallery_" + tabname + " textarea");

            // check masks_a, masks_b are available
            if (masks_a.value == "" && masks_b.value == "") {
                let id = selected_gallery_index();
                let src = gradioApp().querySelector("#generation_info_" + tabname + " textarea").value;
                if (src) {
                    let infotexts = JSON.parse(src).infotexts;
                    if (infotexts[id]) {
                        parse_generation_info(tabname, infotexts[id]);
                    }
                }
            }

            var lightbox = gradioApp().getElementById("lightboxModal");
            var lightbox_wrap = lightbox.querySelector(".mudd_masks_wrapper");
            var gallery_id = tabname + "_gallery";
            var wrap = gradioApp().querySelectorAll("#" + gallery_id + " .mudd_masks_wrapper")[0];

            // toggle mask visibility
            if (lightbox_wrap) {
                if (lightbox_wrap?.style.top == "0px") {
                    lightbox_wrap.style.top = "-2000px";
                    if (wrap)
                        wrap.style.top = "-2000px";
                } else if (lightbox_wrap) {
                    lightbox_wrap.style.top = "0px";
                    if (wrap)
                        wrap.style.top = "0px";
                }
            }
        });
        gradioApp().querySelector("#image_buttons_" + tabname + " .form").appendChild(maskbtn);
    }
}

function get_current_gallery() {
    let tabs = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery');
    for (let i = 0; i < tabs.length; i++) {
        if (tabs[i].parentElement.offsetParent) {
            return tabs[i];
        }
    }
    return null;
}

function onGalleryUpdate() {
    if (!opts.mudd_use_gallery_detection_preview)
        return;

    if (Object.keys(generation_buttons).length < 2) {
        let button = gradioApp().getElementById('txt2img' + "_generation_info_button");
        if (button)
            generation_buttons.txt2img = button;
        button = gradioApp().getElementById('img2img' + "_generation_info_button");
        if (button)
            generation_buttons.img2img = button;
    }
    if (Object.keys(generation_buttons).length > 0 && generation_buttons.txt2img && !enabled.txt2img) {
        enabled.txt2img = true;
        attachGalleryButtonListener("txt2img");
    }
    if (Object.keys(generation_buttons).length > 0 && generation_buttons.img2img && !enabled.img2img) {
        enabled.img2img = true;
        attachGalleryButtonListener("img2img");
    }

    // check canvas visibility
    var id = selected_gallery_index();
    if (id == -1) {
        var lightbox = gradioApp().getElementById("lightboxModal");
        var lightbox_wrap = lightbox?.querySelector(".mudd_masks_wrapper");
        // toggle mask visibility
        if (lightbox_wrap) {
            lightbox_wrap.style.top = "-2000px";
        }

        var gallery = get_current_gallery();
        if (gallery) {
            var wrap = gallery.querySelector(".mudd_masks_wrapper");
            if (wrap)
                wrap.style.top = "-2000px";
        }
    }
}

onAfterUiUpdate(() => {
    if (opts.mudd_use_gallery_detection_preview)
        onGalleryUpdate();
});

onUiLoaded(()=>{
    // attach mask toggle button
    attach_mask_button();
});

})();
