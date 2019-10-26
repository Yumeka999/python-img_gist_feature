from img_gist_feature.utils_gist import *

s_img_url = "./test/A.jpg"
gist_helper = GistUtils()
np_img = preproc_img(s_img_url)
np_gist = gist_helper.get_gist_vec(np_img)

print(np_gist)