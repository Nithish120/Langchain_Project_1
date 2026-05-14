loader = TextLoader("Repository/RAG/blog.txt", encoding="UTF-8")
#Creates a loader object that points to your file. Doesn't read it yet — just sets it up.

document = loader.load()
#Actually reads the file and returns a list of Document objects. Each Document has:
#page_content → the actual text
#metadata → file path, etc.

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#Creates a splitter that will break text into chunks of 1000 characters with no overlap between chunks.


text = text_splitter.split_documents(document)
#Takes your loaded document and splits it into multiple smaller Document objects.


