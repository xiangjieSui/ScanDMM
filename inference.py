from argparse import ArgumentParser

from pyro.infer import Predictive
from models import DMM
from suppor_lib import *


class Inference():
    def __init__(self, model, img_path, n_scanpaths, length, output_path, if_plot=False):
        self.dmm = model
        self.img_path = img_path
        self.n_scanpaths = n_scanpaths
        self.length = length
        self.output_path = output_path
        self.if_plot = if_plot

    def create_random_starting_points(self, num_points):
        # randomly sampling starting points from an equator bias map
        y, x = [], []

        for i in range(num_points):
            while True:
                temp = np.random.normal(loc=0, scale=0.2)

                # if the sampled y ranges in [-1, 1]
                if (temp <= 1) and (temp >= -1):
                    y.append(temp)
                    break

            # sampling x
            x.append(np.random.uniform(-1, 1))

        cords = np.vstack((np.array(y) * 90, np.array(x) * 180)).swapaxes(0, 1)
        cords = sphere2xyz(torch.from_numpy(cords))

        return cords

    def summary(self, samples):
        # reorganize predictions
        obs = None

        for index in range(int(len(samples) / 2)):
            name = 'obs_x_' + str(index + 1)

            # convert predictions to standard 3D coordinates (x, y, z), where x^2 + y^2+ z^2 = 1
            temp = samples[name].reshape([-1, 3])
            its_sum = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
            temp = temp / torch.unsqueeze(its_sum, 1)

            # convert (x, y, z) to (lat, lon)
            if obs is not None:
                obs = torch.cat((obs, torch.unsqueeze(xyz2plane(temp), dim=0)), dim=0)
            else:
                obs = torch.unsqueeze(xyz2plane(temp), dim=0)

        # let ``n_scanpaths'' to be the first dim
        obs = torch.transpose(obs, 0, 1)

        return obs

    def predict(self):
        # set num_samples larger if your GPU/CPU is out of memory
        # e.g., by setting num_sample = 2 (default = 1), we might only need 1/2 memory.
        # HOWEVER, increasing num_samples resulting a little longer time for prediction.
        # adjust the parameters: num_samples, n_scanpaths to satisfy your situation.
        num_samples = 1
        rep_num = self.n_scanpaths // num_samples
        predictive = Predictive(self.dmm.model, num_samples=num_samples)

        for _, _, files in os.walk(self.img_path):
            num_img = len(files)
            count = 0
            for img in files:
                count += 1
                img_path = os.path.join(self.img_path, img)
                image_tensor = torch.unsqueeze(image_process(img_path), dim=0).repeat([rep_num, 1, 1, 1])
                starting_points = torch.unsqueeze(
                    self.create_random_starting_points(rep_num), dim=1).to(torch.float32)
                _scanpaths = starting_points.repeat([1, self.length, 1])

                # the element in test_mask = 0 if the required length is reached,
                # e.g., [1 1 1 1 0 0...] means producing a 4-second scanpath (noting the max length = self.length);
                # here we consistently produce scanpaths with a length of self.length;
                # modify the test_mask if you want to produce variable-length scanpaths.
                test_mask = torch.ones([rep_num, self.length])

                test_batch = _scanpaths.cuda()
                test_batch_mask = test_mask.cuda()
                test_batch_images = image_tensor.cuda()

                # run model
                with torch.no_grad():
                    samples = predictive(scanpaths=test_batch,
                                         scanpaths_reversed=None,
                                         mask=test_batch_mask,
                                         scanpath_lengths=None,
                                         images=test_batch_images,
                                         predict=True)

                    # scanpaths.shape = [n_scanpaths, n_length, 2]
                    scanpaths = self.summary(samples).cpu().numpy()

                    print('[{}]/[{}]:{} {} scanpaths are produced\nSaving to {}'
                          .format(count, num_img, img, scanpaths.shape[0], self.output_path))
                    save_name = img.split('.')[0] + '.npy'
                    np.save(os.path.join(self.output_path, save_name), scanpaths)

                    if self.if_plot:
                        # plot 20 scanpaths
                        print('Begin to plot scanpaths')

                        length_tensor = (torch.ones(self.n_scanpaths) * self.length).int()

                        if not os.path.exists(self.output_path):
                            os.makedirs(self.output_path)

                        plot_scanpaths(scanpaths, img_path, length_tensor.numpy(), save_path=self.output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='ScanDMM')
    parser.add_argument('--model', default='./model/model_lr-0.0003_bs-64_epoch-435.pkl', type=str,
                        help='model path, default = ./model/model_lr-0.0003_bs-64_epoch-435.pkl')
    parser.add_argument('--inDir', default='./demo/input', type=str,
                        help='image path, default = ./demo/input')
    parser.add_argument('--outDir', default='./demo/output', type=str,
                        help='output path, default = ./demo/output')
    parser.add_argument('--n_scanpaths', default=200, type=int,
                        help='number of produced scanpaths, default = 200')
    parser.add_argument('--length', default=20, type=int,
                        help='length of produced scanpaths, default = 20')
    parser.add_argument('--if_plot', default=True, type=bool,
                        help='plot scanpaths or not, default = True')
    args = parser.parse_args()

    dmm = DMM(use_cuda=config.use_cuda)
    dmm.load_state_dict(torch.load(args.model))

    mytest = Inference(model=dmm,
                       img_path=args.inDir,
                       n_scanpaths=args.n_scanpaths,
                       length=args.length,
                       output_path=args.outDir,
                       if_plot=args.if_plot)
    mytest.predict()
