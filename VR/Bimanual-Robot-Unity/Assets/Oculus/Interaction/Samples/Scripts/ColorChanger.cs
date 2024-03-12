/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.Samples
{
    public class ColorChanger : MonoBehaviour
    {
        [SerializeField]
        private Renderer _target;

        private Material _targetMaterial;
        private Color _savedColor;
        private float _lastHue = 0;

        public void NextColor()
        {
            _lastHue = (_lastHue + 0.3f) % 1f;
            Color newColor = Color.HSVToRGB(_lastHue, 0.8f, 0.8f);
            _targetMaterial.color = newColor;
        }

        public void Save()
        {
            _savedColor = _targetMaterial.color;
        }

        public void Revert()
        {
            _targetMaterial.color = _savedColor;
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(_target);
            _targetMaterial = _target.material;
            Assert.IsNotNull(_targetMaterial);
            _savedColor = _targetMaterial.color;
        }

        private void OnDestroy()
        {
            Destroy(_targetMaterial);
        }
    }
}
