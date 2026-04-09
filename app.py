import os
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from io import BytesIO

st.set_page_config(page_title="Juggernaut v6 Image Generator", layout="centered")
st.title("🎨 Juggernaut v6 Image Generator")

HF_TOKEN = "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"
MODEL_ID = "stablediffusionapi/juggernaut-v6"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            token=HF_TOKEN,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        else:
            pipe.enable_attention_slicing()

        return pipe, device

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


with st.spinner("Loading Juggernaut v6 model (first run may take a while)..."):
    pipe, device = load_pipeline()

if pipe is None:
    st.stop()

st.success(f"Model loaded on **{device.upper()}**")

with st.form("generate_form"):
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="A majestic lion standing on a rocky cliff at sunset, photorealistic, 8k",
        height=100,
    )
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        value="ugly, blurry, low quality, distorted, deformed, watermark, text",
        height=68,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider("Inference Steps", min_value=20, max_value=60, value=30, step=5)
    with col2:
        guidance = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
    with col3:
        seed = st.number_input("Seed (-1 = random)", min_value=-1, max_value=2**31 - 1, value=-1)

    col4, col5 = st.columns(2)
    with col4:
        width = st.selectbox("Width", [512, 640, 768], index=0)
    with col5:
        height = st.selectbox("Height", [512, 640, 768], index=0)

    submitted = st.form_submit_button("🖼️ Generate Image", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        try:
            generator = None
            if seed != -1:
                generator = torch.Generator(device=device).manual_seed(int(seed))

            with st.spinner("Generating image..."):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    height=height,
                    width=width,
                )
                image = result.images[0]

            st.image(image, caption="Generated Image", use_container_width=True)

            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png",
                use_container_width=True,
            )

        except torch.cuda.OutOfMemoryError:
            st.error("GPU out of memory. Try reducing image size or inference steps.")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
