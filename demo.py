import numpy as np
import time

z = np.loadtxt('z.txt')
mask = np.loadtxt('mask.txt')
mask = np.array(mask, dtype=bool)

for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):

                # 操作1：判断条件并将结果存储在 mask 中
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                
                # 操作2：计算值并将结果存储在 z 中
                sdf,_ =self.eval_points(pnts.to(device), all_planes, decoders)
                z.append(sdf.cpu().numpy())


            # 合并结果
            mask = np.concatenate(mask, axis=0)
            z = np.concatenate(z, axis=0)

            # 使用布尔索引将不满足条件的元素设置为 -1
            start_time = time.time()
            z[~mask] = -1 
            end_time = time.time()
            print("!!! -1:",end_time-start_time, "s")
            print("z:", len(z), z) # 329868000
            print("mask:", len(mask), mask) # 329868000