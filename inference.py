from pyro.infer import Predictive
import time
from models import DMM
from suppor_lib import *


class Inference():
    def __init__(self, model, img_path, n_scanpaths, length, output_path):
        self.dmm = model
        self.img_path = img_path
        self.n_scanpaths = n_scanpaths
        self.length = length
        self.output_path = output_path

    def create_random_starting_points(self, num_points):
        y, x = [], []
        for i in range(num_points):
            while True:
                temp = np.random.normal(loc=0, scale=0.2)
                if (temp <= 1) and (temp >= -1):
                    y.append(temp)
                    break
            x.append(np.random.uniform(-1, 1))
        cords = np.vstack((np.array(y) * 90, np.array(x) * 180)).swapaxes(0, 1)
        cords = sphere2xyz(torch.from_numpy(cords))
        return cords

    def summary(self, samples):
        obs = None
        for index in range(int(len(samples) / 2)):
            name = 'obs_x_' + str(index + 1)
            temp = samples[name].reshape([-1, 3])
            its_sum = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
            temp = temp / torch.unsqueeze(its_sum, 1)
            if obs is not None:
                obs = torch.cat((obs, torch.unsqueeze(xyz2plane(temp), dim=0)), dim=0)
            else:
                obs = torch.unsqueeze(xyz2plane(temp), dim=0)
        obs = torch.transpose(obs, 0, 1)
        return obs

    def predict(self):
        # set num_samples larger to produce more scanpaths (n_scanpaths * num_sample) without costing GPU memory
        # e.g., by setting num_sample = 2, we can produce n_scanpaths * 2 scanpaths.
        # HOWEVER, increasing num_samples will case a longer time for prediction.
        # adjust the parameters: num_samples, n_scanpaths to satisfy your situation.
        predictive = Predictive(self.dmm.model, num_samples=1)

        for _, _, files in os.walk(self.img_path):
            num_img = len(files)
            count = 0
            for img in files:
                count += 1
                times = [time.time()]
                img_path = os.path.join(self.img_path, img)
                image_tensor = torch.unsqueeze(image_process(img_path), dim=0).repeat([self.n_scanpaths, 1, 1, 1])
                starting_points = torch.unsqueeze(
                    self.create_random_starting_points(self.n_scanpaths), dim=1).to(torch.float32)
                _scanpaths = starting_points.repeat([1, self.length, 1])

                # the element in test_mask = 0 if the required length is reached,
                # e.g., [1 1 1 1 0 0...] means producing a 4-second scanpath (noting the max length = self.length);
                # here we consistently produce scanpaths with a length of self.length;
                # modify the length_tensor and test_mask if you want to produce variable-length scanpaths.
                length_tensor = (torch.ones(self.n_scanpaths) * self.length).int()
                test_mask = torch.ones([self.n_scanpaths, self.length])

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
                    times.append(time.time())
                    time_cost = times[-1] - times[-2]

                    print('[{}]/[{}]:{} {} scanpaths are produced\t(time cost = {:.3f} sec)\nSaving to {}'
                          .format(count, num_img, img, scanpaths.shape[0], time_cost, self.output_path))
                    save_name = img.split('.')[0] + '.npy'
                    np.save(os.path.join(self.output_path, save_name), scanpaths)

                    print('Begin to plot scanpaths')

                    # plot 20 scanpaths
                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                    plot_scanpaths(scanpaths, img_path, length_tensor.numpy(), save_path=self.output_path)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dmm = DMM(use_cuda=True)
    dmm.load_state_dict(torch.load('./model/lev_lr-0.0003_bs-64_dy-0.99998_epo-601_seed-1234.pkl'))

    mytest = Inference(model=dmm,
                       img_path='./demo/input',
                       n_scanpaths=100,
                       length=20,
                       output_path='./demo/output')
    mytest.predict()