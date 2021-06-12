from django.shortcuts import render
import requests
import sys
from subprocess import run, PIPE
from django.core.files.storage import FileSystemStorage

def button(request):

    return render(request, 'home.html')


def output(request):
    data = requests.get("https://www.google.com/")
    print(data.text)
    data = data.text
    return render(request, 'home.html', {'data': data})


# external script execution
def external(request):
    inp = request.POST.get('param')
    image=request.FILES['image']
    fs=FileSystemStorage()
    filename = fs.save(image.name, image)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    #section-1
    # out = run([sys.executable, '/media/vaibhav/Local_Disk3/open_source_contribution/image2/external_script_for_django/tst.py', inp], shell=False, stdout=PIPE)
    #section-2
    image = run([sys.executable, 'script.py',str(fileurl), str(filename)], shell=False, stdout=PIPE)
    
    gurl = image.stdout
    gurl = gurl.decode("utf-8")
    print(gurl)
    return render(request, 'home.html', { 'raw_url': templateurl, 'edit_url': gurl})
