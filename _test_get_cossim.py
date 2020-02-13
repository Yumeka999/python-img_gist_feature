import json
from img_gist_feature.utils_gist import *
from img_gist_feature.utils_tool import *
import matplotlib.pyplot as plt
import cv2


def get_img_gist_feat(s_img_url):
    gist_helper = GistUtils()
    np_img = cv2.imread(s_img_url, -1)
    np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
    np_gist_L2Norm = np_l2norm(np_gist)
    print()
    print("img url: %s" % s_img_url) 
    print("shape ", np_gist.shape)
    print("gist feature noly show 10dim", np_gist[0,:10], "...")
    print("gist feature(L2 norm) noly show 10dim", np_gist_L2Norm[0,:10], "...")
    print()

    return np_gist_L2Norm

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    np_img_gist_a = get_img_gist_feat(s_img_url_a)
    np_img_gist_b = get_img_gist_feat(s_img_url_b)

    f_img_sim = np.inner(np_img_gist_a, np_img_gist_b)
    print("%.23f" % f_img_sim)


    np_img_group = cv2.imread(s_img_url_a)
    np_img_public = cv2.imread(s_img_url_b)


    fig = plt.figure()
    plt.suptitle("%.7f" % f_img_sim, fontsize=20)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np_img_group[:,:,::-1])
    ax.set_title("%s" % s_img_url_a, fontsize=20)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np_img_public[:,:,::-1])
    ax.set_title("%s" % s_img_url_b, fontsize=20)

    fig.savefig("./test/show.jpg")

    plt.show()


if __name__ == "__main__":
    O_IN = {}
    O_IN['s_img_url_a'] = "./test/A.jpg"
    O_IN['s_img_url_b'] = "./test/B.jpg"
    proc_main(O_IN)

