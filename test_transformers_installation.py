import sys

print('#'*100)
print('## Using the following python executable:')
print(sys.executable)


print('#'*100)
print('## Testing general transformer import')
try:
    import transformers
except:
    print('ERROR: Failed to import transformers')
    sys.exit(1)
print('## OK')
print('#'*100)
print('## Testing import of custom model')
try:
    from transformers import SmallTransformerModel, SmallTransformerConfig
    model = SmallTransformerModel(SmallTransformerConfig())
except:
    print('ERROR: Failed to import custom model')
    sys.exit(1)
print('## OK')

print('#'*100)
