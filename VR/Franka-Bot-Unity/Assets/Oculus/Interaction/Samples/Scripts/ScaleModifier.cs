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

namespace Oculus.Interaction.Samples
{
    public class ScaleModifier : MonoBehaviour
    {
        public void SetScaleX(float x)
        {
            transform.localScale = new Vector3(x, transform.localScale.y, transform.localScale.z);
        }
        public void SetScaleY(float y)
        {
            transform.localScale = new Vector3(transform.localScale.x, y, transform.localScale.z);
        }
        public void SetScaleZ(float z)
        {
            transform.localScale = new Vector3(transform.localScale.x, transform.localScale.y, z);
        }
    }
}
