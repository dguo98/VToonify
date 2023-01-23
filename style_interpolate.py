import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2


class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image/video")
        self.parser.add_argument("--style_id", type=int, default=9, help="the id of the style image")
        self.parser.add_argument("--end_style_id", type=int, default=7, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--video", action="store_true", help="if true, video stylization; if false, image stylization")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--parsing_map_path", type=str, default=None, help="path of the refined parsing map of the target video")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
if __name__ == "__main__":

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    os.makedirs(args.output_path, exist_ok=True)
    
    
    device = "cpu" if args.cpu else "cuda"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
    vtoonify = VToonify(backbone = args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args.style_encoder_path, device)    

    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]

        end_stylename = list(exstyles.keys())[args.end_style_id]

        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():  
            exstyle = vtoonify.zplus2wplus(exstyle)

        end_exstyle = torch.tensor(exstyles[end_stylename]).to(device)
        with torch.no_grad():  
            end_exstyle = vtoonify.zplus2wplus(end_exstyle)


    if args.video and args.parsing_map_path is not None:
        x_p_hat = torch.tensor(np.load(args.parsing_map_path))          
            
    print('Load models successfully!')
    
    
    filename = args.content
    basename = os.path.basename(filename).split('.')[0]
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
    print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
    assert args.video is True
    if args.video:
        cropname = os.path.join(args.output_path, basename + '_input.mp4')
        savename = os.path.join(args.output_path, basename + '_vtoonify_' +  args.backbone[0] + '.mp4')
        
        # HACK (demi)
        #video_cap = cv2.VideoCapture(filename)
        #num = int(video_cap.get(7))
        video_cap = cv2.VideoCapture(filename)
        num = 0
        while True:
            success, frame = video_cap.read()
            if success == False:
                break
            num += 1
        print("nums = ", num)
        video_cap = cv2.VideoCapture(filename)



        # end_style_ratio =  (1.0*i) / num

        first_valid_frame = True
        batch_frames, batch_s_w = [], []
        last_frame, last_I = None, None
        print("filenmae =", filename, " first valid True=", first_valid_frame)
        print("num=",num)
        for i in tqdm(range(num)):
            success, frame = video_cap.read()
            if success == False:
                assert('load video frames error')
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print(f"frame {i} cvtcolor error, reusing the previous frame")
                frame = last_frame
            last_frame = frame

            # We proprocess the video by detecting the face in the first frame, 
            # and resizing the frame so that the eye distance is 64 pixels.
            # Centered on the eyes, we crop the first frame to almost 400x400 (based on args.padding).
            # All other frames use the same resizing and cropping parameters as the first frame.
            if first_valid_frame:
                if args.scale_image:
                    paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
                    if paras is None:
                        continue
                    h,w,top,bottom,left,right,scale = paras
                    H, W = int(bottom-top), int(right-left)
                    # for HR video, we apply gaussian blur to the frames to avoid flickers caused by bilinear downsampling
                    # this can also prevent over-sharp stylization results. 
                    if scale <= 0.75:
                        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                    if scale <= 0.375:
                        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
                else:
                    H, W = frame.shape[0], frame.shape[1]
                print("first valid frame")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                videoWriter = cv2.VideoWriter(cropname, fourcc, video_cap.get(5), (W, H))
                videoWriter2 = cv2.VideoWriter(savename, fourcc, video_cap.get(5), (4*W, 4*H))
                
                # For each video, we detect and align the face in the first frame for pSp to obtain the style code. 

                first_valid_frame = False
            elif args.scale_image:
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

            # This style code is used for all other frames.
            with torch.no_grad():
                try:
                    I = align_face(frame, landmarkpredictor)
                except:
                    print(f"frame {i} detection error, reusing the previous frame")
                    cv2.imwrite(f"{args.output_path}/detect_bad_{i}.jpg", frame)
                    I = last_I
                last_I = I

                I = transform(I).unsqueeze(dim=0).to(device)
                s_w = pspencoder(I)
                s_w = vtoonify.zplus2wplus(s_w)
                
                end_style_ratio = (1.0 * i) / num
                cur_exstyle = end_exstyle * end_style_ratio + exstyle * (1-end_style_ratio)
                if vtoonify.backbone == 'dualstylegan':
                    if args.color_transfer:
                        s_w = cur_exstyle
                    else:
                        s_w[:,:7] = cur_exstyle[:,:7]
                batch_s_w.append(s_w)

            videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            batch_frames += [transform(frame).unsqueeze(dim=0).to(device)]

            if len(batch_frames) == args.batch_size or (i+1) == num:
                x = torch.cat(batch_frames, dim=0)
                s_w = torch.cat(batch_s_w, dim=0)
                batch_frames, batch_s_w = [], []
                with torch.no_grad():
                    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
                    # followed by downsampling the parsing maps
                    if args.video and args.parsing_map_path is not None:
                        x_p = x_p_hat[i+1-x.size(0):i+1].to(device)
                    else:
                        x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                        scale_factor=0.5, recompute_scale_factor=False).detach()
                    # we give parsing maps lower weight (1/16)
                    inputs = torch.cat((x, x_p/16.), dim=1)
                    # d_s has no effect when backbone is toonify
                    y_tilde = vtoonify(inputs, s_w, d_s = args.style_degree)       
                    y_tilde = torch.clamp(y_tilde, -1, 1)
                for k in range(y_tilde.size(0)):
                    videoWriter2.write(tensor2cv2(y_tilde[k].cpu()))
        videoWriter.release()
        videoWriter2.release()
        video_cap.release()
        
        finalname = savename.replace(".mp4", "_audio.mp4")
        os.system(f"ffmpeg -i {savename} -i {args.content} -c:v copy -c:a aac {finalname} -max_muxing_queue_size 9999")

    
    print('Transfer style successfully!')