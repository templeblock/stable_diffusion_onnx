import os
import inspect
import numpy as np
import onnxruntime as ort
# tokenizer
from transformers import CLIPTokenizer
# utils
from tqdm import tqdm
from diffusers import LMSDiscreteScheduler, PNDMScheduler
import cv2

# 'CUDAExecutionProvider', 'CPUExecutionProvider', 'TensorrtExecutionProvider',
ORT_EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ORT_EP_LIST_CPU = ['CPUExecutionProvider']


class StableDiffusionEngine:
    def __init__(
            self,
            scheduler,
            model="stable-diffusion-v1-5",
            tokenizer="openai/clip-vit-large-patch14",
            device="CPU"
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.scheduler = scheduler

        self.text_encoder_in_shape0 = [1, 77]
        self.vae_encoder_in_shape0 = [1, 3, 512, 512]
        self.vae_decoder_in_shape0 = [1, 4, 64, 64]
        self.unet_inshape0 = [1, 4, 64, 64]

        self.model_path_text_encoder = os.path.join(model, "text_encoder/model.onnx")
        self.model_path_unet = os.path.join(model, "unet/model.onnx")
        self.model_path_vae_encoder = os.path.join(model, "vae_encoder/model.onnx")
        self.model_path_vae_decoder = os.path.join(model, "vae_decoder/model.onnx")

        self.sess_text_encoder = None
        self.sess_unet = None
        self.sess_vae_encoder = None
        self.sess_vae_decoder = None

        self.latent_shape = self.unet_inshape0[1:]
        self.init_image_shape = self.vae_encoder_in_shape0[2:]

    def run_text_encoder(self, input_data):
        """
        name: input_ids
        type: int32[batch,sequence]

        name: last_hidden_state
        type: float32[Addlast_hidden_state_dim_0,Addlast_hidden_state_dim_1,768]
        name: pooler_output
        type: float32[Reshapepooler_output_dim_0,Reshapepooler_output_dim_1]
        """
        if self.sess_text_encoder is None:
            self.sess_text_encoder = ort.InferenceSession(self.model_path_text_encoder, providers=ORT_EP_LIST)

        out_names = ["last_hidden_state", "pooler_output"]
        outputs = self.sess_text_encoder.run(out_names, {'input_ids': input_data.astype("int32")})
        return outputs[0]

    def run_unet(self, latent_model_input, timestep, encoder_hidden_states):
        '''
        name: sample
        type: float32[batch,channels,height,width]
        name: timestep
        type: int64[batch]
        name: encoder_hidden_states
        type: float32[batch,sequence,768]

        name: out_sample
        type: float32[Convout_sample_dim_0,Convout_sample_dim_1,Convout_sample_dim_2,Convout_sample_dim_3]
        '''
        if self.sess_unet is None:
            # use gpu when memory is enough
            self.sess_unet = ort.InferenceSession(self.model_path_unet, providers=ORT_EP_LIST_CPU)

        out_names = ["out_sample"]
        in_names = ["sample", "timestep", "encoder_hidden_states"]
        in_datas = [latent_model_input, timestep, encoder_hidden_states]
        outputs = self.sess_unet.run(out_names, dict(zip(in_names, in_datas)))
        return outputs[0]

    def run_vae_encoder(self, input_data):
        """
        sample
        name: sample
        type: float32[batch,channels,height,width]

        name: latent_sample
        type: float32[Addlatent_sample_dim_0,Addlatent_sample_dim_1,Addlatent_sample_dim_2,Addlatent_sample_dim_3]
        """
        if self.sess_vae_encoder is None:
            self.sess_vae_encoder = ort.InferenceSession(self.model_path_vae_encoder, providers=ORT_EP_LIST)

        input_data = np.array(input_data).astype("float32")

        out_names = ["latent_sample"]
        outputs = self.sess_vae_encoder.run(out_names, {'sample': input_data})
        return outputs[0]

    def run_vae_decoder(self, input_data):
        """
        name: latent_sample
        type: float32[batch,channels,height,width]

        name: sample
        type: float32[Convsample_dim_0,Convsample_dim_1,Convsample_dim_2,Convsample_dim_3]
        """
        if self.sess_vae_decoder is None:
            self.sess_vae_decoder = ort.InferenceSession(self.model_path_vae_decoder, providers=ORT_EP_LIST)

        input_data = np.array(input_data).astype("float32")

        out_names = ["sample"]
        outputs = self.sess_vae_decoder.run(out_names, {'latent_sample': input_data})
        return outputs[0]

    def _preprocess_mask(self, mask):
        h, w = mask.shape
        if h != self.init_image_shape[0] and w != self.init_image_shape[1]:
            mask = cv2.resize(
                mask,
                (self.init_image_shape[1], self.init_image_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        mask = cv2.resize(
            mask,
            (self.init_image_shape[1] // 8, self.init_image_shape[0] // 8),
            interpolation=cv2.INTER_NEAREST
        )
        mask = mask.astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)
        mask = 1 - mask
        return mask

    def _preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[1:]
        if h != self.init_image_shape[0] and w != self.init_image_shape[1]:
            image = cv2.resize(
                image,
                (self.init_image_shape[1], self.init_image_shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        # to batch
        image = image[None].transpose(0, 3, 1, 2)
        return image

    def _encode_image(self, init_image):
        moments = self.run_vae_encoder(self._preprocess_image(init_image))

        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def __call__(
            self,
            prompt,
            init_image=None,
            mask=None,
            strength=0.5,
            num_inference_steps=32,
            guidance_scale=7.5,
            eta=0.0
    ):
        # extract condition
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids

        print("tokens:", tokens)

        text_embeddings = self.run_text_encoder(np.array([tokens]))

        # do classifier free guidance
        if guidance_scale > 1.0:
            tokens_uncond = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            ).input_ids

            uncond_embeddings = self.run_text_encoder(np.array([tokens_uncond]))
            text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # initialize latent latent
        if init_image is None:
            latents = np.random.randn(*self.latent_shape)
            init_timestep = num_inference_steps
        else:
            init_latents = self._encode_image(init_image)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = np.array([[self.scheduler.timesteps[-init_timestep]]]).astype(np.long)
            noise = np.random.randn(*self.latent_shape)
            latents = self.scheduler.add_noise(init_latents, noise, timesteps)[0]

        if init_image is not None and mask is not None:
            mask = self._preprocess_mask(mask)
        else:
            mask = None

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.stack([latents, latents], 0) if guidance_scale > 1.0 else latents[None]
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            latent_model_input = latent_model_input.astype("float32")
            text_embeddings = text_embeddings.astype("float32")

            # predict the noise residual
            noise_pred = self.run_unet(latent_model_input, np.array([t]).astype("int64"), text_embeddings)

            # perform guidance
            if guidance_scale > 1.0:
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            # masking for inapinting
            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents, noise, t)
                latents = ((init_latents_proper * mask) + (latents * (1 - mask)))[0]

        # release resource
        self.sess_text_encoder = None
        self.sess_unet = None
        self.sess_vae_encoder = None

        print("begin run_vae_decoder")
        image = self.run_vae_decoder(np.expand_dims(latents, 0))

        # convert tensor to opencv's image format
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        return image
